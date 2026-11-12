"""Validated configuration and structured logging for ComplianceGuard.

Settings are loaded from ``config.yaml`` and may be overridden through
``CG_*`` environment variables. The API key comes only from the
environment (``OPENAI_API_KEY``); without it the application runs in demo
mode instead of failing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TextIO, Tuple, Type

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFIG_ENV_VAR = "CG_CONFIG_FILE"
LOGGER_ROOT = "cg"


class _Section(BaseModel):
    """Base for nested sections: immutable, unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSettings(_Section):
    name: str = Field(default="gpt-4o", min_length=1)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, ge=16, le=131072)


class ApiSettings(_Section):
    timeout: float = Field(default=60.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=5, ge=0, le=10)
    base_url: str = "https://api.openai.com/v1"
    json_mode: bool = True
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    retry_initial_wait: float = Field(default=1.0, ge=0.0, le=60.0)
    retry_max_wait: float = Field(default=30.0, ge=0.0, le=600.0)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value


class EvaluationSettings(_Section):
    fuzzy_match_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    min_quote_chars: int = Field(default=12, ge=1, le=200)
    concurrency: int = Field(default=5, ge=1, le=64)
    checkpoint_path: str = "results_partial.jsonl"
    output_path: str = "results.jsonl"
    max_cost_usd: Optional[float] = Field(default=None, gt=0.0)


class PromptSettings(_Section):
    """Optional overrides; built-in English prompts are used when unset."""

    system_prompt_path: Optional[str] = None
    user_template_path: Optional[str] = None


class AppSettings(_Section):
    """Dashboard safety limits."""

    max_upload_mb: float = Field(default=5.0, gt=0.0, le=100.0)
    max_cases: int = Field(default=500, ge=1, le=10000)
    data_dir: str = "data"
    default_budget_usd: float = Field(default=5.0, gt=0.0)


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = True


class PricingEntry(_Section):
    """USD per one million tokens."""

    input_per_1m: float = Field(ge=0.0)
    output_per_1m: float = Field(ge=0.0)


def _default_pricing() -> Dict[str, PricingEntry]:
    return {
        "gpt-4o": PricingEntry(input_per_1m=2.50, output_per_1m=10.00),
        "gpt-4o-mini": PricingEntry(input_per_1m=0.15, output_per_1m=0.60),
    }


def _config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)


class Settings(BaseSettings):
    """Top-level settings (see config.yaml for the documented defaults)."""

    model_config = SettingsConfigDict(
        env_prefix="CG_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        populate_by_name=True,  # Settings(openai_api_key=...) despite the alias
    )

    model: ModelSettings = Field(default_factory=ModelSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    pricing: Dict[str, PricingEntry] = Field(default_factory=_default_pricing)
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "CG_OPENAI_API_KEY"),
        repr=False,
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(
            settings_cls, yaml_file=_config_path())
        return (init_settings, env_settings, dotenv_settings,
                yaml_source, file_secret_settings)

    @property
    def demo_mode(self) -> bool:
        """True when no API key is configured (mocks are used)."""
        return self.openai_api_key is None

    def price_for(self, model_name: str) -> Optional[PricingEntry]:
        """Exact match first, then the longest matching name prefix.

        ``gpt-4o-mini-2024-07-18`` resolves to ``gpt-4o-mini``, not
        ``gpt-4o``.
        """
        if model_name in self.pricing:
            return self.pricing[model_name]
        matches = [name for name in self.pricing
                   if model_name.startswith(name)]
        if not matches:
            return None
        return self.pricing[max(matches, key=len)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings; raises ValidationError on bad config."""
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

class Status(str, Enum):
    """Outcome label required on every log line."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARSE_ERROR = "PARSE_ERROR"


_FIELD_KEYS = ("case_id", "model", "tokens_used", "status", "error_reason")


class JsonFormatter(logging.Formatter):
    """One JSON object per line; mandatory fields are always present."""

    def format(self, record: logging.LogRecord) -> str:
        fields: Dict[str, Any] = dict(getattr(record, "cg", {}))
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key in _FIELD_KEYS:
            payload[key] = fields.pop(key, None)
        payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PlainFormatter(logging.Formatter):
    """Human-readable single-line format with the mandatory fields."""

    def format(self, record: logging.LogRecord) -> str:
        fields: Dict[str, Any] = dict(getattr(record, "cg", {}))
        head = "%s %-7s %s %s" % (
            datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"),
            record.levelname, record.name, record.getMessage())
        tail = " ".join("%s=%s" % (k, fields.pop(k, None))
                        for k in _FIELD_KEYS)
        extra = " ".join("%s=%s" % (k, v) for k, v in fields.items())
        line = "%s | %s%s" % (head, tail, (" " + extra) if extra else "")
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``cg`` namespace."""
    return logging.getLogger("%s.%s" % (LOGGER_ROOT, name))


def configure_logging(level: str = "INFO", json_format: bool = True,
                      stream: Optional[TextIO] = None) -> logging.Logger:
    """Idempotent: replaces the handler on the ``cg`` logger."""
    root = logging.getLogger(LOGGER_ROOT)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if json_format else PlainFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))
    root.propagate = False
    return root


def log_event(logger: logging.Logger, level: int, event: str, *,
              case_id: Optional[str] = None, model: Optional[str] = None,
              tokens_used: Optional[int] = None,
              status: Optional[Status] = None,
              error_reason: Optional[str] = None,
              **extra: Any) -> None:
    """Log ``event`` with the mandatory fields (None when not applicable)."""
    fields: Dict[str, Any] = {
        "case_id": case_id,
        "model": model,
        "tokens_used": tokens_used,
        "status": status.value if isinstance(status, Status) else status,
        "error_reason": error_reason,
    }
    fields.update(extra)
    logger.log(level, event, extra={"cg": fields})