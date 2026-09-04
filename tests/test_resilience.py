"""Retry, backoff, and the two-layer error classification (PLAN §9.2, §9.3, §9.6)."""
import pytest

from app.observability.metrics import RunMetrics
from app.resilience import (
    EMPTY_CODES,
    PARTIAL_USABLE_CODES,
    RETRYABLE_CODES,
    ProviderError,
    backoff_delay,
    classify_response,
    classify_status_code,
    retry_with_backoff,
)
from app.tools.mock import MockBackend

from tests.conftest import DOMAIN


def _envelope(top: int, task: int, *, results=None) -> dict:
    return {
        "status_code": top, "status_message": "Ok." if top == 20000 else "error",
        "tasks": [{"status_code": task, "status_message": "msg", "result": results}],
    }


@pytest.mark.parametrize("code,expected", [
    (20000, "success"),
    (40102, "empty"),           # "No Search Results." -- a success with zero rows
    (40106, "partial_usable"),  # partial results, still worth keeping
    (40101, "retryable"),
    (40103, "retryable"),
    (40202, "retryable"),
    (40209, "retryable"),
    (50000, "retryable"),
    (50301, "retryable"),
    (50401, "retryable"),
    (20100, "terminal"),        # "Task Created." -- invalid for the live endpoints
    (40100, "terminal"),
    (40200, "terminal"),
    (50100, "terminal"),        # 5xxxx but permanent; a range check would get this wrong
    (None, "terminal"),
])
def test_status_code_classification(code, expected):
    assert classify_status_code(code) == expected


def test_ranges_would_be_wrong():
    """The two cases that make broad 4xxxx/5xxxx ranges incorrect."""
    assert classify_status_code(40101) == "retryable"   # inside the "4xxxx = client" range
    assert classify_status_code(50100) == "terminal"    # inside the "5xxxx = retry" range
    assert RETRYABLE_CODES & (EMPTY_CODES | PARTIAL_USABLE_CODES) == set()


def test_task_level_failure_inside_http_200():
    """DataForSEO returns most failures inside a 200; the top level says Ok."""
    classification, code, message = classify_response(_envelope(20000, 50000))
    assert classification == "retryable" and code == 50000
    assert "task" in message


def test_task_level_empty_is_not_a_failure():
    classification, code, _ = classify_response(_envelope(20000, 40102))
    assert (classification, code) == ("empty", 40102)


def test_task_level_partial_keeps_rows():
    payload = _envelope(20000, 40106, results=[{"items": [{"type": "organic"}]}])
    classification, _, _ = classify_response(payload)
    assert classification == "partial_usable"
    assert payload["tasks"][0]["result"][0]["items"], "partial rows must be preserved"


def test_top_level_terminal_short_circuits():
    classification, code, _ = classify_response(_envelope(40200, 20000))
    assert (classification, code) == ("terminal", 40200)


def test_backoff_is_exponential_and_capped():
    plain = [backoff_delay(i, 0.5, 8.0, jitter=False) for i in range(6)]
    assert plain == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]


def test_full_jitter_stays_within_the_window():
    for attempt in range(5):
        ceiling = min(8.0, 0.5 * 2 ** attempt)
        assert 0.0 <= backoff_delay(attempt, 0.5, 8.0, jitter=True) <= ceiling


def test_terminal_error_is_never_retried():
    calls = []

    def boom():
        calls.append(1)
        raise ProviderError("nope", classification="terminal", status_code=40200)

    with pytest.raises(ProviderError):
        retry_with_backoff(boom, attempts=4, base_delay=0, max_delay=0, sleep=lambda _: None)
    assert len(calls) == 1


def test_retry_then_succeed(make_run):
    state = make_run(backend=MockBackend(domain_hint=DOMAIN,
                                         fail_first_n={"google_serp": 2}))
    metrics = RunMetrics(state["node_events"]).as_dict()

    assert state["report_document"]["status"] == "completed"
    assert state["errors"] == []
    assert metrics["total_retries"] == 2, "two failures then success"
    assert metrics["nodes"]["retrieve"]["retries"] == 2


def test_retries_exhausted_degrades_without_crashing(make_run):
    """Every call fails permanently: the run still produces a report."""
    state = make_run(backend=MockBackend(
        domain_hint=DOMAIN,
        fail_first_n={"google_serp": 99, "keyword_metrics": 99, "chatgpt_response": 99},
    ))
    document = state["report_document"]

    assert document["status"] == "partial"
    assert state["degraded"] is True
    assert len(state["errors"]) == len(state["tool_calls"]), "every call failed"
    assert all(e.classification == "retryable" for e in state["errors"])
    assert all(e.attempts == 3 for e in state["errors"])
    assert state["normalized"] == []
    # every planned query still gets a row, marked failed -- see test_persistence
    assert document["queries"] and all(q["retrieval_status"] == "failed"
                                       for q in document["queries"])
    assert document["summary"], "a report is produced even when nothing was retrieved"
