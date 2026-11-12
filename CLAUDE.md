# ComplianceGuard — AI Agent Auditor

## Architecture
- `main.py` — orchestrator, CLI, batch processing.
- `validator.py` — JSON extraction, fuzzy citation verification. DO NOT simplify citation checks; they are our core value.
- `core/metrics.py` — single source of truth for all metrics (Epistemic Failure, etc.).
- `app.py` — Streamlit dashboard (reads results.jsonl).
- `data/*.jsonl` — adversarial test cases.

## Critical Engineering Principles
1. Evidence-based auditing is the core product functionality. Never weaken evidence verification to improve apparent detection rates.
2. Every credited finding must pass citation validation. A finding with `cited_step_ok != True` must never be counted as valid.
3. Never silently modify the output schema, validator behavior, evaluation methodology, or existing metrics.
4. Preserve backward compatibility unless an explicit migration is designed and approved.
5. Never modify ground-truth labels simply to make test results look better.
6. Distinguish between actual policy violations, suspicious behavior, and uncertain cases.
7. Do not introduce mock results into production evaluation.
8. Before implementing a new feature, inspect the existing architecture and identify all affected components.
9. Run regression tests after every meaningful change.
10. If a proposed change weakens audit reliability, explain the risk before implementing it.

## Strict Constraints
- Do NOT add LangChain, databases, or auth yet. We are in MVP phase.
- After any code change, run `pytest` and `python main.py --data data/demo_case.jsonl --model gpt-4o`.