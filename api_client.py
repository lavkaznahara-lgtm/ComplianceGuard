"""Asynchronous LLM client for ComplianceGuard.

* ``OpenAILLMClient`` - ``openai.AsyncOpenAI`` with tenacity exponential
  backoff, tiktoken token counting and per-call USD cost.
* ``MockLLMClient``   - deterministic simulated auditor for Demo Mode.
* ``create_client``   - picks the real client when ``OPENAI_API_KEY`` is
  set and degrades to the mock (with a warning) when it is not.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config import Settings, Status, get_logger, log_event
from validator import normalize_ground_truth

ChatMessage = Mapping[str, str]
_LOG = get_logger("api_client")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base error; ``reason`` is safe to log."""

    def __init__(self, reason: str, *, status_code: Optional[int] = None,
                 attempts: int = 1) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.attempts = attempts


class LLMAuthError(LLMError):
    """Missing or rejected credentials (never retried)."""


class LLMRequestError(LLMError):
    """The request itself is invalid (never retried)."""


class LLMRetriesExhausted(LLMError):
    """A transient error persisted through every retry."""


# ---------------------------------------------------------------------------
# Token counting and cost
# ---------------------------------------------------------------------------

class TokenCounter:
    """tiktoken-backed counter with an explicit estimate fallback.

    tiktoken downloads its encodings on first use. In an offline or
    restricted environment that fails; the counter then falls back to a
    ``ceil(chars / 4)`` estimate and says so once (``exact`` is False).
    """

    TOKENS_PER_MESSAGE = 3
    REPLY_PRIMING = 3

    def __init__(self, model: str, *, offline: bool = False) -> None:
        self._model = model
        self._encoding: Any = None
        if not offline:
            self._encoding = self._load(model)

    @staticmethod
    def _load(model: str) -> Any:
        try:
            import tiktoken
        except ImportError:
            _LOG.warning("tiktoken is not installed; token counts are estimates")
            return None
        try:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                return tiktoken.get_encoding("o200k_base")
        except Exception as exc:  # network or cache failure
            _LOG.warning(
                "tiktoken encoding unavailable (%s: %s); falling back to a "
                "chars/4 estimate", type(exc).__name__, str(exc)[:80])
            return None

    @property
    def exact(self) -> bool:
        return self._encoding is not None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return max(1, math.ceil(len(text) / 4))

    def count_messages(self, messages: Sequence[ChatMessage]) -> int:
        total = self.REPLY_PRIMING
        for message in messages:
            total += self.TOKENS_PER_MESSAGE
            total += self.count_text(str(message.get("role", "")))
            total += self.count_text(str(message.get("content", "")))
        return total


def compute_cost(settings: Settings, model: str, input_tokens: int,
                 output_tokens: int) -> Tuple[float, bool]:
    """Return ``(usd, priced)``; ``priced`` is False for unknown models."""
    price = settings.price_for(model)
    if price is None:
        return 0.0, False
    usd = (input_tokens * price.input_per_1m
           + output_tokens * price.output_per_1m) / 1_000_000.0
    return usd, True


@dataclass(frozen=True)
class LLMResponse:
    """One completed call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    attempts: int
    token_source: str  # api | tiktoken | estimate
    finish_reason: Optional[str] = None
    simulated: bool = False

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostTracker:
    """Accumulates usage across calls (safe within one event loop)."""

    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    simulated: bool = False

    def record(self, response: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cost_usd += response.cost_usd
        self.simulated = self.simulated or response.simulated

    def record_failure(self) -> None:
        self.failed_calls += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "simulated": self.simulated,
        }


@runtime_checkable
class LLMClient(Protocol):
    """Interface shared by the real and the mock client."""

    simulated: bool
    tracker: CostTracker

    async def complete(self, messages: Sequence[ChatMessage], *,
                       case_id: str) -> LLMResponse: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------

def _is_quota_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if code == "insufficient_quota":
        return True
    return isinstance(body, dict) and (
        body.get("code") == "insufficient_quota"
        or (isinstance(body.get("error"), dict)
            and body["error"].get("code") == "insufficient_quota"))


def is_retryable(exc: BaseException) -> bool:
    """Transient failures only: timeouts, connection drops, 429, 408, 5xx."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return not _is_quota_error(exc)  # an empty balance will not heal
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 429) or exc.status_code >= 500
    return False


class OpenAILLMClient:
    """``AsyncOpenAI`` wrapper; the SDK's own retries are disabled so that
    tenacity is the only retry layer."""

    simulated = False

    def __init__(self, settings: Settings, *,
                 client: Optional[AsyncOpenAI] = None) -> None:
        self._settings = settings
        if client is None:
            key = settings.openai_api_key
            if key is None:
                raise LLMAuthError("OPENAI_API_KEY is not set")
            client = AsyncOpenAI(
                api_key=key.get_secret_value(),
                base_url=settings.api.base_url,
                timeout=settings.api.timeout,
                max_retries=0,
            )
        self._client = client
        self._counter = TokenCounter(settings.model.name)
        self._warned_unpriced = False
        self.tracker = CostTracker()

    def _request_kwargs(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        cfg = self._settings
        kwargs: Dict[str, Any] = {
            "model": cfg.model.name,
            "messages": [dict(m) for m in messages],
            cfg.api.token_param: cfg.model.max_tokens,
        }
        if cfg.model.temperature is not None:
            kwargs["temperature"] = cfg.model.temperature
        if cfg.api.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _before_sleep(self, case_id: str) -> Any:
        model = self._settings.model.name

        def log_retry(state: RetryCallState) -> None:
            exc = state.outcome.exception() if state.outcome else None
            wait = state.next_action.sleep if state.next_action else 0.0
            log_event(
                _LOG, logging.WARNING, "llm_retry", case_id=case_id,
                model=model, tokens_used=None, status=Status.FAIL,
                error_reason="%s: %s" % (type(exc).__name__, str(exc)[:120]),
                attempt=state.attempt_number, next_wait_s=round(wait, 2))
        return log_retry

    async def complete(self, messages: Sequence[ChatMessage], *,
                       case_id: str) -> LLMResponse:
        cfg = self._settings
        model = cfg.model.name
        kwargs = self._request_kwargs(messages)
        attempts = 0
        started = time.perf_counter()
        completion: Any = None
        try:
            async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(cfg.api.max_retries + 1),
                    wait=wait_exponential_jitter(
                        initial=cfg.api.retry_initial_wait,
                        max=cfg.api.retry_max_wait),
                    retry=retry_if_exception(is_retryable),
                    before_sleep=self._before_sleep(case_id),
                    reraise=True):
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    completion = await self._client.chat.completions.create(
                        **kwargs)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise self._fail(LLMAuthError, case_id, exc, attempts) from exc
        except openai.APIStatusError as exc:
            if is_retryable(exc):
                raise self._fail(LLMRetriesExhausted, case_id, exc,
                                 attempts) from exc
            raise self._fail(LLMRequestError, case_id, exc, attempts) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise self._fail(LLMRetriesExhausted, case_id, exc,
                             attempts) from exc
        except openai.OpenAIError as exc:
            raise self._fail(LLMError, case_id, exc, attempts) from exc

        choices = getattr(completion, "choices", None)
        if not choices:
            raise self._fail(LLMError, case_id,
                             ValueError("response has no choices"), attempts)
        content = choices[0].message.content or ""
        finish = getattr(choices[0], "finish_reason", None)
        input_tokens, output_tokens, source = self._usage(
            completion, messages, content)
        cost, priced = compute_cost(cfg, model, input_tokens, output_tokens)
        if not priced and not self._warned_unpriced:
            self._warned_unpriced = True
            log_event(_LOG, logging.WARNING, "model_not_priced", case_id=case_id,
                      model=model, tokens_used=input_tokens + output_tokens,
                      status=Status.PASS,
                      error_reason="no pricing entry; cost reported as 0.00")
        response = LLMResponse(
            content=content, model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_s=time.perf_counter() - started, attempts=attempts,
            token_source=source, finish_reason=finish)
        self.tracker.record(response)
        log_event(
            _LOG, logging.INFO, "llm_call", case_id=case_id, model=model,
            tokens_used=response.tokens_used, status=Status.PASS,
            error_reason="output truncated (finish_reason=length)"
            if finish == "length" else None,
            cost_usd=round(cost, 6), latency_s=round(response.latency_s, 3),
            attempts=attempts, token_source=source)
        return response

    def _usage(self, completion: Any, messages: Sequence[ChatMessage],
               content: str) -> Tuple[int, int, str]:
        """Prefer the provider's usage block; verify/fill with tiktoken."""
        usage = getattr(completion, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None)
        generated = getattr(usage, "completion_tokens", None)
        if isinstance(prompt, int) and isinstance(generated, int):
            return prompt, generated, "api"
        source = "tiktoken" if self._counter.exact else "estimate"
        return (self._counter.count_messages(messages),
                self._counter.count_text(content), source)

    def _fail(self, error_cls: type, case_id: str, exc: BaseException,
              attempts: int) -> LLMError:
        reason = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        self.tracker.record_failure()
        log_event(_LOG, logging.ERROR, "llm_call_failed", case_id=case_id,
                  model=self._settings.model.name, tokens_used=None,
                  status=Status.FAIL, error_reason=reason, attempts=attempts)
        status_code = getattr(exc, "status_code", None)
        error = error_cls(reason, status_code=status_code, attempts=attempts)
        assert isinstance(error, LLMError)
        return error

    async def aclose(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------------
# Demo-mode client
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Deterministic simulated auditor (no network, no key, no real cost).

    With a ``gold`` registry (case id -> case dict) it imitates a
    fallible auditor: mostly right, sometimes partial, hallucinated,
    missing, or confidently wrong. Output follows the real response
    schema, so the whole pipeline (parsing, citation checks, metrics)
    runs unchanged. Without a registry it answers ``clean``.
    Everything it reports is flagged ``simulated``.
    """

    simulated = True

    def __init__(self, settings: Settings, *,
                 gold: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 latency_s: float = 0.02, seed: int = 7) -> None:
        self._settings = settings
        self._gold: Mapping[str, Mapping[str, Any]] = gold or {}
        self._latency = latency_s
        self._seed = seed
        self._counter = TokenCounter(settings.model.name, offline=True)
        self.tracker = CostTracker(simulated=True)

    async def complete(self, messages: Sequence[ChatMessage], *,
                       case_id: str) -> LLMResponse:
        started = time.perf_counter()
        await asyncio.sleep(self._latency)
        payload = self._respond(case_id, self._gold.get(case_id))
        content = json.dumps(payload, ensure_ascii=False)
        input_tokens = self._counter.count_messages(messages)
        output_tokens = self._counter.count_text(content)
        model = self._settings.model.name
        cost, _ = compute_cost(self._settings, model, input_tokens, output_tokens)
        response = LLMResponse(
            content=content, model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_s=time.perf_counter() - started, attempts=1,
            token_source="estimate", finish_reason="stop", simulated=True)
        self.tracker.record(response)
        log_event(_LOG, logging.DEBUG, "llm_call_simulated", case_id=case_id,
                  model=model, tokens_used=response.tokens_used,
                  status=Status.PASS, error_reason=None, simulated=True)
        return response

    async def aclose(self) -> None:
        return None

    # -- simulation ---------------------------------------------------------

    def _respond(self, case_id: str,
                 case: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if case is None:
            return {"verdict": "clean", "findings": [],
                    "recommended_action": "No action required."}
        rng = random.Random("%s:%s" % (self._seed, case_id))
        roll = rng.random()
        label = str(case.get("label", ""))
        if label == "violation":
            return self._violation(case, rng, roll)
        if label == "ambiguous":
            return self._ambiguous(case, rng, roll)
        return self._clean(case, rng, roll)

    @staticmethod
    def _gold_findings(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
        categories = list(case.get("violation_categories") or [])
        return [{"category": g.category, "step": g.step, "quote": g.quote}
                for g in normalize_ground_truth(case.get("ground_truth"),
                                                categories) if g.quote]

    @staticmethod
    def _finding(base: Mapping[str, Any], confidence: str,
                 severity: str = "high") -> Dict[str, Any]:
        return {"category": base["category"], "step": base["step"],
                "quote": base["quote"],
                "explanation": "Simulated finding for demo mode.",
                "severity": severity, "confidence": confidence}

    def _violation(self, case: Mapping[str, Any], rng: random.Random,
                   roll: float) -> Dict[str, Any]:
        gold = self._gold_findings(case)
        conf = "high" if rng.random() < 0.7 else "medium"
        if not gold or roll >= 0.82:  # missed entirely
            return {"verdict": "clean", "findings": [],
                    "recommended_action": "No action required."}
        if roll < 0.58:  # complete and correct
            findings = [self._finding(g, conf) for g in gold]
        elif roll < 0.72:  # right verdict, second category missed
            findings = [self._finding(gold[0], conf)]
        else:  # right verdict, hallucinated citation
            findings = [self._finding(
                {**gold[0], "quote": "The agent knowingly disclosed the "
                                     "customer records to an outside party."},
                conf)]
        return {"verdict": "violation", "findings": findings,
                "recommended_action": "Escalate to the security team."}

    def _ambiguous(self, case: Mapping[str, Any], rng: random.Random,
                   roll: float) -> Dict[str, Any]:
        gold = self._gold_findings(case)
        base = gold[0] if gold else self._any_action(case)
        if base is None or roll >= 0.85:
            return {"verdict": "clean", "findings": [],
                    "recommended_action": "No action required."}
        if roll < 0.35:
            return {"verdict": "uncertain",
                    "findings": [self._finding(base, "low", "medium")],
                    "recommended_action": "Route to a human reviewer."}
        if roll < 0.55:
            return {"verdict": "suspicious",
                    "findings": [self._finding(base, "medium", "medium")],
                    "recommended_action": "Route to a human reviewer."}
        return {"verdict": "violation",
                "findings": [self._finding(base, "high")],
                "recommended_action": "Escalate to the security team."}

    def _clean(self, case: Mapping[str, Any], rng: random.Random,
               roll: float) -> Dict[str, Any]:
        if roll < 0.80:
            return {"verdict": "clean", "findings": [],
                    "recommended_action": "No action required."}
        base = self._any_action(case)
        if base is None:
            return {"verdict": "clean", "findings": [],
                    "recommended_action": "No action required."}
        if roll < 0.92:  # confident false alarm
            return {"verdict": "violation",
                    "findings": [self._finding(base, "high")],
                    "recommended_action": "Escalate to the security team."}
        return {"verdict": "suspicious",
                "findings": [self._finding(base, "medium", "low")],
                "recommended_action": "Route to a human reviewer."}

    @staticmethod
    def _any_action(case: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        steps = case.get("log")
        if not isinstance(steps, list):
            return None
        for step in steps:
            if step.get("type") == "action" and len(str(step.get("content", ""))) > 20:
                text = str(step["content"])
                return {"category": "SCOPE_VIOLATION", "step": step.get("step"),
                        "quote": text[:60].rsplit(" ", 1)[0] if len(text) > 60 else text}
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(settings: Settings, *,
                  gold: Optional[Mapping[str, Mapping[str, Any]]] = None,
                  force_demo: bool = False,
                  strict_live: bool = False) -> LLMClient:
    """Real client if a key is configured, otherwise the mock.

    Default: never raises for a missing key; it logs a warning and returns
    the simulated client so the dashboard keeps working.

    ``strict_live=True`` is for real evaluations: a missing key raises
    ``LLMAuthError`` and the simulated client is never returned. ``gold``
    is only ever handed to the simulated client, never to the real one.
    """
    if strict_live and force_demo:
        raise ValueError("strict_live and force_demo are mutually exclusive")
    if strict_live:
        if settings.demo_mode:
            log_event(_LOG, logging.ERROR, "live_mode_unavailable", case_id=None,
                      model=settings.model.name, tokens_used=None,
                      status=Status.FAIL,
                      error_reason="OPENAI_API_KEY is not set (strict live mode)")
            raise LLMAuthError(
                "OPENAI_API_KEY is not set; strict live mode will not fall "
                "back to the simulated client")
        return OpenAILLMClient(settings)
    if force_demo or settings.demo_mode:
        reason = ("demo mode requested" if force_demo
                  else "OPENAI_API_KEY is not set")
        log_event(_LOG, logging.WARNING, "demo_mode_enabled", case_id=None,
                  model=settings.model.name, tokens_used=None,
                  status=Status.FAIL, error_reason=reason,
                  detail="results are simulated; no API calls are made")
        return MockLLMClient(settings, gold=gold)
    return OpenAILLMClient(settings)