"""Tests for JSON extraction and audit-response normalisation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator import extract_json, parse_audit_response  # noqa: E402

CLEAN = {"verdict": "clean", "findings": [], "recommended_action": "none"}


@pytest.mark.parametrize("text, method", [
    (json.dumps(CLEAN), "direct"),
    ("```json\n" + json.dumps(CLEAN) + "\n```", "fenced"),
    ("```\n" + json.dumps(CLEAN) + "\n```", "fenced"),
    ("Sure! Here is the audit:\n```json\n" + json.dumps(CLEAN)
     + "\n```\nLet me know if you need more.", "fenced"),
    ("Result: " + json.dumps(CLEAN) + " (end)", "balanced"),
    ("\ufeff  " + json.dumps(CLEAN, indent=2) + "  ", "direct"),
])
def test_extracts_clean_object(text: str, method: str) -> None:
    result = extract_json(text)
    assert result.data == CLEAN
    assert result.method == method
    assert result.error is None


def test_braces_inside_strings_do_not_break_extraction() -> None:
    payload = {"verdict": "violation", "findings": [
        {"category": "SCOPE_VIOLATION", "step": 3, "quote": "format {x} and }"}]}
    result = extract_json("Output -> " + json.dumps(payload) + " <- done")
    assert result.data == payload
    assert result.method == "balanced"


def test_prefers_the_object_that_has_a_verdict() -> None:
    text = 'noise {"a": 1} then ' + json.dumps(CLEAN) + ' and {"b": 2}'
    result = extract_json(text)
    assert result.data == CLEAN


def test_trailing_comma_is_repaired() -> None:
    result = extract_json('{"verdict": "clean", "findings": [],}')
    assert result.data == {"verdict": "clean", "findings": []}
    assert result.method == "repaired"


def test_unescaped_inner_quotes_are_repaired() -> None:
    text = ('{"verdict": "violation", "findings": [{"category": '
            '"DATA_LEAKAGE", "step": 14, "quote": "He said "ok" then left", '
            '"confidence": "high"}]}')
    result = extract_json(text)
    assert result.method == "repaired"
    assert result.data is not None
    finding = result.data["findings"][0]
    assert finding["quote"] == 'He said "ok" then left'
    assert finding["confidence"] == "high"


def test_smart_quotes_as_delimiters_are_repaired() -> None:
    text = "{\u201cverdict\u201d: \u201cclean\u201d, \u201cfindings\u201d: []}"
    result = extract_json(text)
    assert result.data == {"verdict": "clean", "findings": []}


def test_raw_newline_inside_string_is_repaired() -> None:
    text = '{"verdict": "clean", "findings": [], "recommended_action": "a\nb"}'
    result = extract_json(text)
    assert result.data is not None
    assert result.data["recommended_action"] == "a\nb"


@pytest.mark.parametrize("text, fragment", [
    ("", "empty"),
    ("   \n  ", "empty"),
    ("I cannot help with that.", "no JSON"),
    ('{"verdict": "violation", "findings": [{"category": "DATA_LE',
     "truncated"),
    ("[1, 2, 3]", "no JSON"),
])
def test_failures_report_a_reason(text: str, fragment: str) -> None:
    result = extract_json(text)
    assert result.data is None
    assert result.method == "none"
    assert result.error is not None and fragment in result.error


@pytest.mark.parametrize("value", [None, 123, b"{}", ["x"]])
def test_non_string_input_never_raises(value: Any) -> None:
    result = extract_json(value)  # type: ignore[arg-type]
    assert result.data is None
    assert result.error == "empty response"


def _parse(payload: Any) -> Any:
    return parse_audit_response(json.dumps(payload))


def test_verdict_is_normalised() -> None:
    assert _parse({"verdict": " VIOLATION ", "findings": []}).verdict == "violation"
    assert _parse({"verdict": "Ambiguous", "findings": []}).verdict == "uncertain"


@pytest.mark.parametrize("payload, reason", [
    ({"findings": []}, "missing verdict"),
    ({"verdict": "maybe", "findings": []}, "invalid verdict"),
    ({"verdict": "clean", "findings": "none"}, "not a list"),
])
def test_invalid_structures_are_rejected(payload: Any, reason: str) -> None:
    parsed = _parse(payload)
    assert parsed.ok is False
    assert parsed.error_reason is not None and reason in parsed.error_reason


def test_missing_findings_means_empty() -> None:
    parsed = _parse({"verdict": "clean"})
    assert parsed.ok and parsed.findings == []


def test_findings_are_normalised_and_unknown_ones_dropped() -> None:
    parsed = _parse({"verdict": "violation", "findings": [
        {"category": "scope violation", "step": "3", "quote": "q",
         "severity": "HIGH", "confidence": "High"},
        {"category": "data-leakage", "step": 2.0, "confidence": "very high",
         "severity": "catastrophic"},
        {"category": "MADE_UP", "step": 1},
        "not an object",
    ]})
    assert parsed.ok
    assert [f.category for f in parsed.findings] == ["SCOPE_VIOLATION",
                                                     "DATA_LEAKAGE"]
    first, second = parsed.findings
    assert first.step == 3 and first.severity == "high"
    assert first.confidence == "high"
    assert second.step == 2
    assert second.confidence is None and second.severity is None
    reasons = sorted(d.reason for d in parsed.dropped)
    assert reasons == ["not_an_object", "unknown_category:MADE_UP"]


@pytest.mark.parametrize("step, expected", [
    (3, 3), ("7", 7), (4.0, 4), (4.5, None), (True, None), ("abc", None),
    (None, None), (-1, -1),
])
def test_step_coercion(step: Any, expected: Optional[int]) -> None:
    parsed = _parse({"verdict": "violation", "findings": [
        {"category": "DECEPTION", "step": step, "quote": "q"}]})
    assert parsed.findings[0].step == expected


def test_parse_failure_propagates_extraction_reason() -> None:
    parsed = parse_audit_response("no json here")
    assert parsed.ok is False
    assert parsed.method == "none"
    assert parsed.error_reason == "no JSON object found in response"