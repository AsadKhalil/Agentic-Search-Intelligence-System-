"""Structured JSON logging with a correlation id carried across the whole DAG run.

Every log line emitted anywhere below an active `bind_run(...)` scope carries the
same `correlation_id`, so one API request can be followed node-by-node with:

    cat logs.ndjson | jq 'select(.correlation_id=="<id>")'
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_run_uuid: ContextVar[str | None] = ContextVar("run_uuid", default=None)
_node: ContextVar[str | None] = ContextVar("node", default=None)

# Anything whose key matches one of these is replaced before it reaches a log sink.
_SENSITIVE_KEYS = {
    "password", "api_key", "openai_api_key", "dataforseo_password",
    "authorization", "token", "secret", "login",
}
_REDACTED = "***redacted***"

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName",
}


def redact(value: Any) -> Any:
    """Recursively blank out sensitive values. Applied to every log payload."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if k.lower() in _SENSITIVE_KEYS else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + f"...<truncated {len(value) - 2000} chars>"
    return value


def redact_field(key: str, value: Any) -> Any:
    """A sensitive name passed as a top-level `extra=` key needs the same treatment as a
    nested one -- redact() alone only sees the value."""
    return _REDACTED if key.lower() in _SENSITIVE_KEYS else redact(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in ((k, v) for k, v in record.__dict__.items() if k not in _RESERVED):
            payload[key] = redact_field(key, value)
        for ctx_key, ctx_var in (
            ("correlation_id", _correlation_id),
            ("run_uuid", _run_uuid),
            ("node", _node),
        ):
            val = ctx_var.get()
            if val and ctx_key not in payload:
                payload[ctx_key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable variant for local development (LOG_FORMAT=console)."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and k != "correlation_id"
        }
        cid = _correlation_id.get() or "-"
        tail = " ".join(f"{k}={redact_field(k, v)}" for k, v in extras.items())
        return f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} [{cid[:8]}] {record.getMessage()} {tail}".rstrip()


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # uvicorn installs its own noisy handlers; let them propagate to ours instead.
    for noisy in ("uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def bind_run(correlation_id: str, run_uuid: str | None = None) -> Iterator[str]:
    tokens = [_correlation_id.set(correlation_id), _run_uuid.set(run_uuid)]
    try:
        yield correlation_id
    finally:
        _correlation_id.reset(tokens[0])
        _run_uuid.reset(tokens[1])


@contextmanager
def bind_node(node: str) -> Iterator[str]:
    token = _node.set(node)
    try:
        yield node
    finally:
        _node.reset(token)


def current_correlation_id() -> str | None:
    return _correlation_id.get()
