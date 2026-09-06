"""End-to-end demo, offline. `make demo`

Runs the pipeline three times against the mock transport -- healthy, a dependency that
fails twice then recovers, and a total outage -- so the retry, degradation and fallback
paths are all visible.

Writes two artifacts:
  demo-output.json   every scenario's full report, metrics, planned calls and errors
  demo-logs.ndjson   the structured log stream, one JSON object per line

Console output stays human-sized; the files carry the detail.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.graph.build import build_graph
from app.graph.state import initial_state
from app.llm import llm_mode
from app.observability.logging import (
    JsonFormatter,
    bind_run,
    configure_logging,
    new_correlation_id,
)
from app.observability.metrics import RunMetrics
from app.schemas import ProfileSnapshot
from app.tools.dataforseo import DataForSEOClient
from app.tools.mock import MockBackend

PROFILE = ProfileSnapshot(
    uuid="demo-profile", name="Acme", domain="acme.io",
    industry="project management", competitors=["asana.com", "monday.com"],
)
QUESTION = "Are we visible for agile planning tools?"

SCENARIOS = [
    ("1. healthy run", None),
    ("2. SERP fails twice then recovers", {"google_serp": 2}),
    ("3. every dependency down",
     {"google_serp": 99, "keyword_metrics": 99, "chatgpt_response": 99}),
]


def run(label: str, fail_first_n: dict[str, int] | None) -> dict[str, Any]:
    settings = get_settings()
    graph = build_graph(
        client=DataForSEOClient(settings, backend=MockBackend(
            domain_hint=PROFILE.domain, fail_first_n=fail_first_n)),
        settings=settings,
    )
    correlation_id = new_correlation_id()
    print(f"\n=== {label}  (correlation_id={correlation_id}) ===")
    with bind_run(correlation_id):
        state = graph.invoke(initial_state(profile=PROFILE, question=QUESTION,
                                           correlation_id=correlation_id))

    document = state["report_document"]
    metrics = RunMetrics(state["node_events"]).as_dict()
    scenario = {
        "label": label,
        "injected_failures": fail_first_n,
        "correlation_id": correlation_id,
        "status": document["status"],
        "degraded": document["degraded"],
        "path": metrics["node_sequence"],
        "totals": {
            "api_calls": metrics["total_api_calls"],
            "retries": metrics["total_retries"],
            "llm_tokens": metrics["total_tokens"],
            "duration_ms": metrics["total_duration_ms"],
        },
        "planned_calls": [{"tool": c.name, "args": c.args} for c in state["tool_calls"]],
        "errors": [e.model_dump() for e in state["errors"]],
        "node_events": [e.model_dump() for e in state["node_events"]],
        "metrics": metrics,
        "report": document,
    }

    # Console gets the shape of the run; the file gets everything.
    print(json.dumps({
        "status": scenario["status"],
        "degraded": scenario["degraded"],
        "path": scenario["path"],
        **scenario["totals"],
        "errors": [
            {"tool": e["tool"] or e["node"], "queries": e["query_keys"],
             "attempts": e["attempts"], "message": e["message"]}
            for e in scenario["errors"]
        ],
        "visibility": document["visibility"],
        "summary": document["summary"],
    }, indent=2))
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-format", default="json", choices=["json", "console"])
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--out", default="demo-output.json",
                        help="results file; pass an empty string to skip")
    parser.add_argument("--log-file", default="demo-logs.ndjson",
                        help="structured log stream; pass an empty string to skip")
    parser.add_argument("--report", default="demo-report.html",
                        help="HTML console built from the results; empty string to skip")
    parser.add_argument("--no-open", action="store_true",
                        help="write the report but do not open a browser")
    args = parser.parse_args()

    configure_logging(args.log_level, args.log_format)
    if args.log_file:
        Path(args.log_file).unlink(missing_ok=True)
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(JsonFormatter())      # always JSON, whatever the console uses
        logging.getLogger().addHandler(handler)

    settings = get_settings()
    artifact: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": QUESTION,
        "profile": PROFILE.model_dump(),
        "config": {
            "llm_mode": llm_mode(settings),
            "llm_model": settings.llm_model,
            # The demo always constructs a MockBackend, whatever MOCK_DATAFORSEO says,
            # so the failure scenarios stay reproducible and cost nothing. Recording the
            # setting alone would imply this artifact came from the live API.
            "dataforseo_transport": "mock fixtures (the demo never calls the live API)",
            "mock_dataforseo_setting": settings.mock_dataforseo,
            "retry": {
                "max_attempts": settings.retry_max_attempts,
                "base_delay_seconds": settings.retry_base_delay_seconds,
                "max_delay_seconds": settings.retry_max_delay_seconds,
                "jitter": settings.retry_jitter,
            },
            "timeouts": {
                "http_seconds": settings.http_timeout_seconds,
                "chatgpt_seconds": settings.chatgpt_timeout_seconds,
            },
        },
        "scenarios": [run(label, fail) for label, fail in SCENARIOS],
    }

    if args.out:
        Path(args.out).write_text(json.dumps(artifact, indent=2, default=str))
        print(f"\nwrote {args.out} "
              f"({Path(args.out).stat().st_size / 1024:.0f} KB, "
              f"{len(artifact['scenarios'])} scenarios)")
    if args.log_file:
        lines = sum(1 for _ in open(args.log_file))
        print(f"wrote {args.log_file} ({lines} log lines)")
        print(f"  jq -c 'select(.correlation_id==\"{artifact['scenarios'][-1]['correlation_id']}\")' "
              f"{args.log_file}")

    # The JSON is the record; the report is the thing anyone actually reads.
    if args.report and args.out:
        import report as report_module

        report_module.build(args.out, args.report, open_after=not args.no_open)


if __name__ == "__main__":
    main()
