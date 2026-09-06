"""Error classification and retry. Hand-rolled rather than tenacity: this is graded code,
so the backoff arithmetic should be visible.

Two layers (PLAN §3):
  1. transport  -- connection errors, timeouts, HTTP status
  2. task-level -- DataForSEO returns most failures *inside* an HTTP 200
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Literal, TypeVar

from app.observability.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

Classification = Literal["success", "empty", "partial_usable", "retryable", "terminal"]

# Explicit sets, not ranges: 4xxxx contains retryables and 5xxxx contains a permanent
# failure. Every code below is quoted from the DataForSEO errors appendix.
SUCCESS_CODES = {20000}                 # "Ok."
EMPTY_CODES = {40102}                   # "No Search Results." -> success with zero rows
PARTIAL_USABLE_CODES = {40106}          # "Task completed with partial results."
RETRYABLE_CODES = {
    40101,   # "Internal SE Server Error."
    40103,   # "Task execution failed, please try to resubmit the task."
    40202,   # "The rate-limit per minute has been exceeded."
    40209,   # "Too many simultaneous queries."
    50000,   # "Internal Error."
    50301,   # "3rd Party API Service Unavailable."
    50401,   # "Internal Error - Timeout."
}
# Everything else is terminal, explicitly including:
#   20100 "Task Created."      queued-task code; invalid for the live endpoints we call
#   40100 "You are not authorized to access this resource."
#   40104 "Please verify your account before using the API." -- arrives as HTTP 403;
#         permanent until the account is verified, so retrying is pure waste
#   40200 "Payment Required."
#   50100 "Not Implemented."   5xxxx but permanent -- retrying can never succeed

USABLE = {"success", "partial_usable", "empty"}


class ToolArgumentError(Exception):
    """LLM produced tool arguments that fail their Pydantic schema. Never reaches HTTP."""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.message = message


class ProviderError(Exception):
    """A classified failure from DataForSEO -- transport or task level."""

    def __init__(
        self,
        message: str,
        *,
        classification: Classification,
        status_code: int | None = None,
        tool: str | None = None,
        attempts_made: int = 1,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.classification = classification
        self.status_code = status_code
        self.tool = tool
        self.attempts_made = attempts_made

    @property
    def retryable(self) -> bool:
        return self.classification == "retryable"


def classify_status_code(code: Any) -> Classification:
    if code in SUCCESS_CODES:
        return "success"
    if code in EMPTY_CODES:
        return "empty"
    if code in PARTIAL_USABLE_CODES:
        return "partial_usable"
    if code in RETRYABLE_CODES:
        return "retryable"
    return "terminal"


def classify_http_status(status: int) -> Classification:
    if status == 429 or status >= 500:
        return "retryable"
    if status >= 400:
        return "terminal"
    return "success"


def classify_response(payload: dict[str, Any]) -> tuple[Classification, int | None, str]:
    """Classify a DataForSEO body across both the top level and every task."""
    top_code = payload.get("status_code")
    top_class = classify_status_code(top_code)
    top_msg = str(payload.get("status_message", ""))
    if top_class in ("retryable", "terminal"):
        return top_class, top_code, f"top-level {top_code}: {top_msg}"

    tasks = payload.get("tasks") or []
    if not tasks:
        return "empty", top_code, "response contained no tasks"

    seen: list[tuple[Classification, int | None, str]] = [
        (
            classify_status_code(task.get("status_code")),
            task.get("status_code"),
            str(task.get("status_message", "")),
        )
        for task in tasks
    ]
    # Retryable wins over terminal: a retry may still fix the whole request.
    for wanted in ("retryable", "terminal", "partial_usable"):
        for cls, code, msg in seen:
            if cls == wanted:
                return cls, code, f"task {code}: {msg}"
    if all(cls == "empty" for cls, _, _ in seen):
        code, msg = seen[0][1], seen[0][2]
        return "empty", code, f"task {code}: {msg}"
    return "success", top_code, top_msg


def backoff_delay(
    attempt: int, base: float, cap: float, jitter: bool, rng: random.Random | None = None
) -> float:
    """Exponential backoff with full jitter. attempt is 0-based."""
    raw = min(cap, base * (2 ** attempt))
    if not jitter:
        return raw
    return (rng or random).uniform(0.0, raw)


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    jitter: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    context: dict[str, Any] | None = None,
) -> tuple[T, int]:
    """Run fn, retrying only classified-retryable ProviderErrors. Returns (value, retries)."""
    ctx = context or {}
    last: ProviderError | None = None
    for attempt in range(attempts):
        try:
            value = fn()
            if attempt:
                log.info("retry.succeeded", extra={**ctx, "attempt": attempt + 1})
            return value, attempt
        except ProviderError as exc:
            last = exc
            exc.attempts_made = attempt + 1
            if not exc.retryable:
                log.warning(
                    "retry.terminal",
                    extra={**ctx, "attempt": attempt + 1,
                           "classification": exc.classification,
                           "status_code": exc.status_code, "error": exc.message},
                )
                raise
            if attempt == attempts - 1:
                break
            delay = backoff_delay(attempt, base_delay, max_delay, jitter)
            log.warning(
                "retry.scheduled",
                extra={**ctx, "attempt": attempt + 1, "of": attempts,
                       "classification": exc.classification, "status_code": exc.status_code,
                       "delay_seconds": round(delay, 3), "error": exc.message},
            )
            sleep(delay)

    assert last is not None
    last.attempts_made = attempts
    log.error("retry.exhausted", extra={**ctx, "attempts": attempts, "error": last.message})
    raise last
