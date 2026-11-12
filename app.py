"""ComplianceGuard dashboard (Streamlit).

Tabs: Run Evaluation, Results Dashboard, Demo Mode. Demo Mode needs no
API key: it shows pre-baked *simulated* results so every chart renders
immediately. Nothing the auditor model or a log file says is ever
rendered as HTML.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped, unused-ignore]
import streamlit as st

import main as pipeline
from api_client import TokenCounter, create_client
from config import (
    PROJECT_ROOT,
    Settings,
    Status,
    configure_logging,
    get_logger,
    get_settings,
    log_event,
)
from core.metrics import (
    OUTCOME_CATEGORIES_OFF,
    OUTCOME_CORRECT,
    OUTCOME_EPISTEMIC,
    OUTCOME_HEDGED,
    OUTCOME_ORDER,
    OUTCOME_SILENT,
    OUTCOME_UNGRADED,
    OUTCOME_UNVERIFIED,
    category_breakdown,
    outcome_class,
    outcome_counts,
)
from validator import ALLOWED_CATEGORIES, normalize_ground_truth, render_log

_LOG = get_logger("app")

Record = Dict[str, Any]
Case = Dict[str, Any]

SOURCE_DEMO_CASE = "Built-in demo case"
SOURCE_SERVER = "Server file (data folder)"
SOURCE_UPLOAD = "Upload a file"
DEMO_LABEL = "DEMO data (120 simulated audit records)"
BUILTIN_CASE = PROJECT_ROOT / "data" / "demo_case.jsonl"

OUTCOME_COLORS = {
    OUTCOME_CORRECT: "#2e9e5b",
    OUTCOME_UNVERIFIED: "#7fb069",
    OUTCOME_CATEGORIES_OFF: "#9bd18b",
    OUTCOME_HEDGED: "#f0c040",
    OUTCOME_SILENT: "#e08a3c",
    OUTCOME_EPISTEMIC: "#c0392b",
    OUTCOME_UNGRADED: "#8a8a8a",
}
VERDICT_COLUMNS = ("clean", "suspicious", "uncertain", "violation", "error")


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit calls; unit-testable)
# ---------------------------------------------------------------------------

def format_usd(value: float) -> str:
    """``$1.23``; four decimals for sub-cent amounts so they are not shown as 0."""
    if 0 < value < 0.01:
        return "$%.4f" % value
    return "${:,.2f}".format(value)


def format_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.1f%%" % value


def verdict_matrix(results: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    """Rows: ground-truth labels; columns: verdicts (+ errors)."""
    rows = []
    for label in pipeline.LABELS:
        row = []
        for column in VERDICT_COLUMNS:
            row.append(sum(
                1 for r in results if r.get("label") == label and (
                    (r.get("verdict") is None and column == "error")
                    or r.get("verdict") == column)))
        rows.append(row)
    return rows


def category_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Per-category detection and over-reporting (same logic as the metrics)."""
    rows = []
    for category, row in category_breakdown(results).items():
        reported = row["detected"] + row["unexpected"]
        rows.append({
            "category": category, "expected": row["expected"],
            "detected": row["detected"], "unexpected": row["unexpected"],
            "recall_%": (round(100.0 * row["detected"] / row["expected"], 1)
                         if row["expected"] else None),
            "precision_%": (round(100.0 * row["detected"] / reported, 1)
                            if reported else None)})
    return pd.DataFrame(rows)


def records_to_frame(results: Sequence[Mapping[str, Any]],
                     blind_ids: Sequence[str]) -> pd.DataFrame:
    blind = set(blind_ids)
    rows = []
    for r in results:
        rows.append({
            "id": r.get("id"),
            "label": r.get("label"),
            "verdict": r.get("verdict") or "-",
            "status": r.get("status"),
            "confidence": r.get("max_conf"),
            "machine_blindness": r.get("id") in blind,
            "outcome": outcome_class(r),
            "expected_categories": ", ".join(r.get("gold_categories") or []),
            "predicted_categories": ", ".join(r.get("pred_categories") or []),
            "unexpected_categories": ", ".join(sorted(
                set(r.get("pred_categories") or []) - set(r.get("gold_categories") or []))),
            "citations_ok": "%s/%s" % (r.get("quotes_ok", 0), r.get("quotes_total", 0)),
            "gt_findings": "%s/%s" % (r.get("gt_findings_matched", 0),
                                      r.get("gt_findings_total", 0)),
            "cost_usd": float(r.get("cost_usd") or 0.0),
            "reason": r.get("error_reason") or "",
        })
    return pd.DataFrame(rows)


def estimate_run_cost(cases: Sequence[Case], settings: Settings,
                      system_prompt: str, user_template: str,
                      output_tokens_per_case: int = 600) -> Optional[float]:
    """Rough upper-bound estimate (offline token estimate); None if unpriced."""
    price = settings.price_for(settings.model.name)
    if price is None:
        return None
    counter = TokenCounter(settings.model.name, offline=True)
    total_in = sum(counter.count_messages(
        pipeline.build_messages(c, system_prompt, user_template)) for c in cases)
    total_out = min(output_tokens_per_case, settings.model.max_tokens) * len(cases)
    return (total_in * price.input_per_1m + total_out * price.output_per_1m) / 1e6


def build_demo_results(settings: Settings, n: int = 120,
                       seed: int = 2026) -> List[Record]:
    """Deterministic SIMULATED audit records in the real result schema."""
    rng = random.Random(seed)
    price = settings.price_for("gpt-4o")
    in_rate = price.input_per_1m if price else 2.5
    out_rate = price.output_per_1m if price else 10.0
    records: List[Record] = []
    for index in range(n):
        label = rng.choices(pipeline.LABELS, weights=(0.22, 0.52, 0.26))[0]
        gold: List[str] = []
        if label == "violation":
            gold = rng.sample(ALLOWED_CATEGORIES, k=rng.choice((1, 1, 2)))
        roll = rng.random()
        verdict: Optional[str]
        predicted: List[str] = []
        confidence = "none"
        quotes_total = quotes_ok = 0
        if label == "violation":
            if roll < 0.58:
                verdict, predicted, confidence = "violation", list(gold), rng.choice(("high", "high", "medium"))
                if rng.random() < 0.15:  # over-claiming: an extra, unexpected category
                    predicted.append(rng.choice([c for c in ALLOWED_CATEGORIES if c not in gold]))
            elif roll < 0.72:
                verdict, predicted, confidence = "violation", gold[:1], rng.choice(("high", "medium"))
            elif roll < 0.82:
                verdict, predicted, confidence = "suspicious", gold[:1], "medium"
            else:
                verdict = "clean"
        elif label == "clean":
            if roll < 0.80:
                verdict = "clean"
            elif roll < 0.92:
                verdict, predicted, confidence = "violation", [rng.choice(ALLOWED_CATEGORIES)], "high"
            else:
                verdict, predicted, confidence = "suspicious", [rng.choice(ALLOWED_CATEGORIES)], "medium"
        else:
            if roll < 0.30:
                verdict, predicted, confidence = "uncertain", [rng.choice(ALLOWED_CATEGORIES)], "low"
            elif roll < 0.50:
                verdict, predicted, confidence = "suspicious", [rng.choice(ALLOWED_CATEGORIES)], "medium"
            elif roll < 0.80:
                verdict, predicted, confidence = "violation", [rng.choice(ALLOWED_CATEGORIES)], "high"
            else:
                verdict = "clean"
        if predicted:
            quotes_total = len(predicted)
            quotes_ok = quotes_total - (1 if rng.random() < 0.08 else 0)
        tokens_in = max(600, int(rng.gauss(2300, 450)))
        tokens_out = max(40, int(rng.gauss(300 if predicted else 60, 80)))
        record: Record = {
            "id": "demo-%03d" % (index + 1), "label": label, "gold_categories": gold,
            "gold_findings": [], "trap_type": None,
            "difficulty": rng.choice(("medium", "hard", "hard", "expert")),
            "security_breach": False, "model": "demo-simulated", "verdict": verdict,
            "pred_categories": predicted, "pred_confidences": [confidence] * len(predicted),
            "quotes_total": quotes_total, "quotes_ok": quotes_ok,
            "findings_total": len(predicted), "invalid_step_findings": 0,
            "max_conf": confidence, "parse_error": False, "parse_error_reason": None,
            "extraction_method": "direct", "findings": [], "finding_errors": [],
            "recommended_action": "",
            "gt_findings_total": len(gold),
            "gt_findings_matched": len(set(gold) & set(predicted)),
            "cost_usd": (tokens_in * in_rate + tokens_out * out_rate) / 1e6,
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "latency_s": round(rng.uniform(0.8, 3.5), 2), "attempts": 1,
            "token_source": "estimate", "simulated": True, "api_error": False,
            "skipped": False,
        }
        records.append(pipeline.finalize_record(record))
    return records


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def accuracy_figure(summary: Mapping[str, Any]) -> go.Figure:
    labels = list(pipeline.LABELS)
    by_label = summary.get("accuracy_by_label", {})
    values = [by_label.get(label, {}).get("accuracy_pct") or 0.0 for label in labels]
    counts = [by_label.get(label, {}).get("n", 0) for label in labels]
    figure = go.Figure(go.Bar(
        x=labels, y=values, marker_color=["#2e9e5b", "#2f6fb5", "#8e6bbf"],
        text=["%.0f%% (n=%d)" % (v, n) for v, n in zip(values, counts)],
        textposition="outside"))
    figure.update_layout(title="Verdict accuracy by ground-truth label",
                         yaxis=dict(title="accuracy (%)", range=[0, 115]),
                         margin=dict(t=50, b=30), height=340)
    return figure


def calibration_figure(epistemic: Mapping[str, Any]) -> go.Figure:
    by_level = epistemic.get("by_level") or {}
    levels = [lv for lv in ("high", "medium", "low") if lv in by_level]
    figure = go.Figure()
    if not levels:
        figure.add_annotation(text="No self-reported confidence in this run",
                              showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    else:
        stated = [by_level[lv]["stated_score"] for lv in levels]
        actual = [by_level[lv]["accuracy"] for lv in levels]
        sizes = [by_level[lv]["n"] for lv in levels]
        figure.add_trace(go.Bar(name="Stated confidence", x=levels, y=stated,
                                marker_color="#2f6fb5"))
        figure.add_trace(go.Bar(name="Actual accuracy", x=levels, y=actual,
                                marker_color="#e08a3c",
                                text=["n=%d" % s for s in sizes], textposition="outside"))
    error = epistemic.get("calibration_error")
    title = "Confidence vs correctness"
    if error is not None:
        title += " (calibration error %.3f)" % error
    figure.update_layout(title=title, barmode="group", height=340,
                         yaxis=dict(title="probability", range=[0, 1.15]),
                         margin=dict(t=50, b=30))
    return figure


def outcome_figure(results: Sequence[Mapping[str, Any]]) -> go.Figure:
    counts = outcome_counts(results)
    names = [n for n in OUTCOME_ORDER if counts[n] > 0]
    figure = go.Figure(go.Pie(
        labels=names, values=[counts[n] for n in names], hole=0.5, sort=False,
        marker=dict(colors=[OUTCOME_COLORS[n] for n in names]),
        textinfo="value+percent"))
    figure.update_layout(title="Outcome breakdown (where confidence misleads)",
                         height=340, margin=dict(t=50, b=10),
                         legend=dict(font=dict(size=10)))
    return figure


def matrix_figure(results: Sequence[Mapping[str, Any]]) -> go.Figure:
    matrix = verdict_matrix(results)
    figure = go.Figure(go.Heatmap(
        z=matrix, x=list(VERDICT_COLUMNS), y=list(pipeline.LABELS),
        text=matrix, texttemplate="%{text}", colorscale="Blues", showscale=False))
    figure.update_layout(title="Ground truth (rows) vs auditor verdict (columns)",
                         height=340, margin=dict(t=50, b=30),
                         yaxis=dict(autorange="reversed"))
    return figure


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

def _log_with_markers(case: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    gold_steps = {f.get("step") for f in record.get("gold_findings") or []}
    pred_steps = {f.get("step") for f in record.get("findings") or []
                  if f.get("credited")}
    lines = []
    for line in render_log(case["log"]).splitlines():
        step = int(line[1:line.index("]")]) if line.startswith("[") else -1
        marker = ("G" if step in gold_steps else " ") + ("P" if step in pred_steps else " ")
        lines.append("%s %s" % (marker, line))
    return "\n".join(lines)


def render_inspector(record: Mapping[str, Any], case: Optional[Mapping[str, Any]],
                     key: str) -> None:
    left, middle, right = st.columns(3)
    left.metric("Ground truth", str(record.get("label")))
    middle.metric("Auditor verdict", str(record.get("verdict") or "-"))
    right.metric("Stated confidence", str(record.get("max_conf")))
    st.caption("Status: %s%s" % (record.get("status"),
                                 " (%s)" % record["error_reason"] if record.get("error_reason") else ""))
    gold = record.get("gold_findings") or []
    if gold:
        st.markdown("**Expected findings (ground truth)**")
        st.dataframe(pd.DataFrame(gold), hide_index=True, key="%s_gold" % key)
    findings = record.get("findings") or []
    if findings:
        st.markdown("**Auditor findings**")
        rows = []
        for f in findings:
            citation = f.get("citation") or {}
            rows.append({
                "category": f.get("category"), "step": f.get("step"),
                "confidence": f.get("confidence"), "credited": f.get("credited"),
                "quote_verified": citation.get("found"),
                "match_score": citation.get("score"), "quote": f.get("quote")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, key="%s_pred" % key)
    if case is not None and case.get("log"):
        st.markdown("**Log** (G = ground-truth step, P = step cited by the auditor)")
        st.code(_log_with_markers(case, record), language="text")


def render_dashboard(results: Sequence[Record], summary: Mapping[str, Any], *,
                     source_label: str, case_index: Mapping[str, Case],
                     key: str) -> None:
    """Shared by the Results Dashboard and Demo Mode tabs."""
    epistemic: Mapping[str, Any] = summary["epistemic"]
    simulated = bool(summary.get("simulated"))
    if simulated:
        st.info("%s. Costs and outcomes are simulated; no API calls were made." % source_label)
    else:
        st.caption(source_label)

    cost_col, cases_col, pass_col, fail_col = st.columns(4)
    cost_col.metric("Total Cost of this Run", format_usd(float(summary["total_cost_usd"])))
    cases_col.metric("Cases audited", "%d" % summary["cases"])
    pass_col.metric("Pass rate", format_pct(summary.get("pass_rate_pct")))
    fail_col.metric("Parse errors / API failures",
                    "%d / %d" % (summary["parse_error"],
                                 sum(1 for r in results if r.get("api_error"))))
    st.caption("Tokens: %s in / %s out" % ("{:,}".format(summary["input_tokens"]),
                                          "{:,}".format(summary["output_tokens"])))

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Epistemic Failure Rate", format_pct(epistemic.get("epistemic_failure_rate_pct")),
              help="Share of graded cases where the auditor was highly confident and wrong or incomplete: wrong verdict, a missing category, or an unexpected one.")
    e2.metric("Calibration error",
              "n/a" if epistemic.get("calibration_error") is None else "%.3f" % epistemic["calibration_error"],
              help="Average gap between stated confidence and actual accuracy.")
    e3.metric("Confidence - correctness delta",
              "n/a" if epistemic.get("confidence_correctness_delta") is None
              else "%+.3f" % epistemic["confidence_correctness_delta"],
              help="Positive means overconfident.")
    e4.metric("Silent misses", format_pct(epistemic.get("silent_miss_rate_pct")),
              help="Wrong verdicts with no stated confidence (for example a missed violation reported as clean).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full accuracy", format_pct(epistemic.get("accuracy_full_pct")),
              help="Right verdict, exactly the expected categories (none missing, none extra) and every quote verified at its cited step.")
    c2.metric("Category precision", format_pct(epistemic.get("category_precision_pct")),
              help="Share of reported categories that were expected.")
    c3.metric("Category recall", format_pct(epistemic.get("category_recall_pct")),
              help="Share of expected categories that were reported.")
    c4.metric("Category Jaccard", format_pct(epistemic.get("category_jaccard_pct")),
              help="Mean overlap between reported and expected categories on violation cases.")

    row1 = st.columns(2)
    row1[0].plotly_chart(accuracy_figure(summary), key="%s_acc" % key)
    row1[1].plotly_chart(calibration_figure(epistemic), key="%s_cal" % key)
    row2 = st.columns(2)
    row2[0].plotly_chart(outcome_figure(results), key="%s_out" % key)
    row2[1].plotly_chart(matrix_figure(results), key="%s_mat" % key)

    st.subheader("Category detection")
    st.dataframe(category_table(results), hide_index=True, key="%s_cat" % key)

    st.subheader("Cases")
    blind_ids: List[str] = sorted(str(i) for i in epistemic.get("machine_blindness_ids") or [])
    frame = records_to_frame(results, blind_ids)
    f1, f2, f3 = st.columns(3)
    only_blind = f1.toggle("Machine-blindness cases only", value=False, key="%s_only" % key,
                           help="High stated confidence and a wrong verdict.")
    picked = f2.multiselect("Filter by machine_blindness_ids", options=blind_ids,
                            key="%s_ids" % key)
    labels = f3.multiselect("Label", options=list(pipeline.LABELS), key="%s_lab" % key)
    f4, f5 = st.columns(2)
    outcomes = f4.multiselect("Outcome", options=list(OUTCOME_ORDER), key="%s_oc" % key)
    needle = f5.text_input("Search id", key="%s_q" % key)

    view = frame
    if only_blind:
        view = view[view["machine_blindness"]]
    if picked:
        view = view[view["id"].isin(picked)]
    if labels:
        view = view[view["label"].isin(labels)]
    if outcomes:
        view = view[view["outcome"].isin(outcomes)]
    if needle.strip():
        view = view[view["id"].astype(str).str.contains(needle.strip(), case=False, regex=False)]
    st.caption("Showing %d of %d cases" % (len(view), len(frame)))
    st.dataframe(view, hide_index=True, key="%s_table" % key,
                 column_config={"cost_usd": st.column_config.NumberColumn("cost_usd", format="$%.4f")})

    if len(view):
        chosen = st.selectbox("Inspect a case", options=list(view["id"]), key="%s_pick" % key)
        record = next(r for r in results if r.get("id") == chosen)
        render_inspector(record, case_index.get(str(chosen)), "%s_insp" % key)

    export = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in results)
    st.download_button("Download results (JSONL)", export, file_name="results.jsonl",
                       mime="application/jsonl", key="%s_dl" % key)


# ---------------------------------------------------------------------------
# Run tab
# ---------------------------------------------------------------------------

def _data_dir(settings: Settings) -> Path:
    path = Path(settings.app.data_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_cases(settings: Settings) -> Tuple[List[Case], List[str]]:
    """Case selection UI. Server files are limited to the data folder."""
    source = st.radio("Case source", [SOURCE_DEMO_CASE, SOURCE_SERVER, SOURCE_UPLOAD],
                      horizontal=True, key="source")
    max_cases = settings.app.max_cases
    cases: List[Case] = []
    problems: List[str] = []
    if source == SOURCE_DEMO_CASE:
        cases, problems = pipeline.load_cases(BUILTIN_CASE, max_cases=max_cases)
    elif source == SOURCE_SERVER:
        folder = _data_dir(settings)
        names = sorted(p.name for p in folder.iterdir()
                       if p.is_file() and p.suffix in (".json", ".jsonl")) if folder.is_dir() else []
        if not names:
            st.warning("No .json/.jsonl files found in %s" % folder.name)
        else:
            name = st.selectbox("File", options=names, key="server_file")
            cases, problems = pipeline.load_cases(folder / name, max_cases=max_cases)
    else:
        uploaded = st.file_uploader("JSONL or JSON file", type=["jsonl", "json"], key="upload")
        limit = int(settings.app.max_upload_mb * 1024 * 1024)
        if uploaded is not None:
            if uploaded.size > limit:
                st.error("File is larger than the %.0f MB limit." % settings.app.max_upload_mb)
            else:
                try:
                    text = uploaded.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    st.error("File must be UTF-8 text.")
                else:
                    cases, problems = pipeline.parse_case_text(text, uploaded.name)
                    if len(cases) > max_cases:
                        problems.append("only the first %d cases are used" % max_cases)
                        cases = cases[:max_cases]
    return cases, problems


async def execute(cases: Sequence[Case], settings: Settings, *, force_demo: bool,
                  budget_usd: float, progress: Any) -> Tuple[List[Record], Dict[str, Any], bool]:
    system_prompt, user_template = pipeline.load_prompts(settings)
    gold = {str(c["id"]): c for c in cases}
    client = create_client(settings, gold=gold, force_demo=force_demo)
    try:
        with tempfile.TemporaryDirectory(prefix="cg_run_") as folder:
            checkpoint = pipeline.Checkpoint(Path(folder) / "results_partial.jsonl", resume=False)
            results = await pipeline.run_batch(
                list(cases), client, settings, system_prompt=system_prompt,
                user_template=user_template, checkpoint=checkpoint,
                max_cost_usd=None if client.simulated else budget_usd, on_progress=progress)
    finally:
        await client.aclose()
    summary = pipeline.summarize(results, client)
    summary["mode"] = "SIMULATED" if client.simulated else "LIVE"
    return results, summary, client.simulated


def render_run_tab(settings: Settings, budget_usd: float) -> None:
    st.subheader("Run an evaluation")
    cases, problems = select_cases(settings)
    if problems:
        with st.expander("%d loading problem(s)" % len(problems)):
            for problem in problems:
                st.text(problem)
    if cases:
        counts = pd.Series([c["label"] for c in cases]).value_counts().to_dict()
        st.write("Loaded **%d** case(s): %s" % (len(cases), ", ".join(
            "%s %d" % (k, v) for k, v in sorted(counts.items()))))

    simulate = st.checkbox("Use the simulated auditor (no API calls, no cost)",
                           value=settings.demo_mode, disabled=settings.demo_mode, key="simulate")
    live = not (simulate or settings.demo_mode)
    if live and cases:
        system_prompt, user_template = pipeline.load_prompts(settings)
        estimate = estimate_run_cost(cases, settings, system_prompt, user_template)
        if estimate is None:
            st.warning("Model %s has no entry in the pricing table: cost will be shown as $0.00." % settings.model.name)
        else:
            st.write("Estimated cost: **up to %s** (rough, assumes 600 output tokens per case)." % format_usd(estimate))
            if estimate > budget_usd:
                st.warning("The estimate exceeds the budget cap of %s; the run stops once spend reaches the cap." % format_usd(budget_usd))
    if settings.demo_mode:
        st.caption("No API key is configured, so only simulated runs are available.")

    if st.button("Run evaluation", type="primary", disabled=not cases, key="run"):
        bar = st.progress(0.0, text="Starting")

        def on_progress(done: int, total: int) -> None:
            bar.progress(done / total, text="Audited %d of %d cases" % (done, total))

        try:
            results, summary, simulated = asyncio.run(execute(
                cases, settings, force_demo=not live, budget_usd=budget_usd, progress=on_progress))
        except Exception as exc:
            log_event(_LOG, logging.ERROR, "ui_run_failed", case_id=None,
                      model=settings.model.name, tokens_used=None, status=Status.FAIL,
                      error_reason="%s: %s" % (type(exc).__name__, str(exc)[:160]))
            st.error("The run failed: %s" % type(exc).__name__)
            return
        st.session_state["results"] = results
        st.session_state["summary"] = summary
        st.session_state["case_index"] = {str(c["id"]): c for c in cases}
        st.session_state["source_label"] = ("SIMULATED run" if simulated else
                                            "Live run with %s" % settings.model.name)
        failed = sum(1 for r in results if r.get("api_error"))
        st.success("Done: %d case(s), total cost %s. Open the Results Dashboard tab."
                   % (len(results), format_usd(float(summary["total_cost_usd"]))))
        if failed:
            st.warning("%d case(s) could not be audited (API errors or the budget cap); "
                       "see the Reason column in the Results Dashboard." % failed)


# ---------------------------------------------------------------------------
# Demo tab
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_demo(n: int, seed: int) -> Tuple[List[Record], Dict[str, Any]]:
    settings = get_settings()
    results = build_demo_results(settings, n=n, seed=seed)
    summary = pipeline.summarize(results)
    summary["mode"] = "DEMO"
    return results, summary


def _load_demo_into_session(results: List[Record], summary: Dict[str, Any]) -> None:
    """Runs as a button callback, i.e. before the script re-renders, so the
    Results Dashboard tab (rendered earlier in the script) already sees it."""
    st.session_state["results"] = results
    st.session_state["summary"] = summary
    st.session_state["case_index"] = {}
    st.session_state["source_label"] = DEMO_LABEL


def render_demo_tab() -> None:
    st.subheader("Demo Mode")
    st.write("Pre-baked **simulated** results, so every chart is visible without an API key. "
             "They illustrate the dashboard and the epistemic metrics; they are not measurements of any real model.")
    results, summary = _cached_demo(120, 2026)
    st.button("Load this demo into the Results Dashboard", key="load_demo",
              on_click=_load_demo_into_session, args=(results, summary))
    if st.session_state.get("source_label") == DEMO_LABEL:
        st.success("Demo data is loaded. Open the Results Dashboard tab.")
    cases, _ = pipeline.load_cases(BUILTIN_CASE, max_cases=1)
    if cases:
        with st.expander("The adversarial demo case: social engineering framed as a compliance audit"):
            case = cases[0]
            st.write("**Policy:** %s" % case.get("policy", ""))
            st.code(render_log(case["log"]), language="text")
            gold = normalize_ground_truth(case.get("ground_truth"),
                                                   list(case.get("violation_categories") or []))
            if gold:
                st.dataframe(pd.DataFrame([g.model_dump() for g in gold]), hide_index=True, key="demo_gold")
    render_dashboard(results, summary, source_label=DEMO_LABEL, case_index={}, key="demo")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def sidebar_settings(settings: Settings) -> Tuple[Settings, float]:
    with st.sidebar:
        st.header("Run settings")
        model = st.text_input("Model", value=settings.model.name, key="model_name").strip()
        threshold = st.slider("Citation match threshold", 0.50, 1.00,
                              float(settings.evaluation.fuzzy_match_threshold), 0.01, key="threshold")
        concurrency = st.slider("Concurrent requests", 1, 16,
                                min(settings.evaluation.concurrency, 16), key="concurrency")
        budget = st.number_input("Budget cap (USD)", min_value=0.01,
                                 value=float(settings.app.default_budget_usd), step=0.5,
                                 format="%.2f", key="budget")
        if settings.demo_mode:
            st.warning("OPENAI_API_KEY is not set: live runs are disabled.")
        else:
            st.success("API key detected (never displayed).")
    updated = settings.model_copy(update={
        "model": settings.model.model_copy(update={"name": model or settings.model.name}),
        "evaluation": settings.evaluation.model_copy(update={
            "fuzzy_match_threshold": threshold, "concurrency": concurrency})})
    return updated, float(budget)


def main() -> None:
    st.set_page_config(page_title="ComplianceGuard", layout="wide")
    base = get_settings()
    configure_logging(base.logging.level, base.logging.json_format)
    st.title("ComplianceGuard")
    st.caption("Epistemic-integrity evaluation for AI support agents")
    settings, budget = sidebar_settings(base)

    run_tab, results_tab, demo_tab = st.tabs(["Run Evaluation", "Results Dashboard", "Demo Mode"])
    with run_tab:
        render_run_tab(settings, budget)
    with results_tab:
        if "results" in st.session_state:
            render_dashboard(st.session_state["results"], st.session_state["summary"],
                             source_label=st.session_state.get("source_label", "Results"),
                             case_index=st.session_state.get("case_index", {}), key="res")
        else:
            st.info("No results yet. Run an evaluation, or open Demo Mode and load the demo data.")
    with demo_tab:
        render_demo_tab()


if __name__ == "__main__":
    main()