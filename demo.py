"""End-to-end demo, offline. `make demo`

Runs the pipeline twice against the mock transport: once cleanly, once with DataForSEO
failing, so the retry, degradation and fallback paths are visible in the log stream.
Every line is structured JSON carrying the run's correlation id.
"""
from __future__ import annotations

import argparse
import json

from app.config import get_settings
from app.graph.build import build_graph
from app.graph.state import initial_state
from app.observability.logging import bind_run, configure_logging, new_correlation_id
from app.observability.metrics import RunMetrics
from app.schemas import ProfileSnapshot
from app.tools.dataforseo import DataForSEOClient
from app.tools.mock import MockBackend

PROFILE = ProfileSnapshot(
    uuid="demo-profile", name="Acme", domain="acme.io",
    industry="project management", competitors=["asana.com", "monday.com"],
)
QUESTION = "Are we visible for agile planning tools?"


def run(label: str, fail_first_n: dict[str, int] | None = None) -> None:
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
    print(json.dumps({
        "status": document["status"],
        "degraded": document["degraded"],
        "path": metrics["node_sequence"],
        "api_calls": metrics["total_api_calls"],
        "retries": metrics["total_retries"],
        "llm_tokens": metrics["total_tokens"],
        "errors": [
            {"tool": e.tool or e.node, "queries": e.query_keys,
             "attempts": e.attempts, "message": e.message}
            for e in state["errors"]
        ],
        "visibility": document["visibility"],
        "top_opportunity": document["queries"][0]["query_key"] if document["queries"] else None,
        "summary": document["summary"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-format", default="json", choices=["json", "console"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level, args.log_format)

    run("1. healthy run")
    run("2. SERP fails twice then recovers", fail_first_n={"google_serp": 2})
    run("3. every dependency down",
        fail_first_n={"google_serp": 99, "keyword_metrics": 99, "chatgpt_response": 99})


if __name__ == "__main__":
    main()
