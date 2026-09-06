"""End-to-end demo, offline. `make demo`

Runs the pipeline three times against the mock transport -- healthy, a dependency that
fails twice then recovers, and a total outage -- so the retry, degradation and fallback
paths are all visible.

Both dependencies are pinned: the mock transport and the deterministic planner. The three
runs differ only in which failures are injected, which is the whole point of the
comparison -- a live planner would re-plan differently each time and cost real money to
say nothing new. Real OpenAI is exercised through the API (`make run`), not here.

Writes two artifacts:
  demo-output.json   every scenario's full report, metrics, planned calls and errors
  demo-logs.ndjson   the structured log stream, one JSON object per line

Console output stays human-sized -- a plain walk-through of one search, then one
summary per run. The structured logs go to the file; `--verbose` also streams them.
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
from app.llm import ScriptedToolCallingLLM, llm_mode
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

# Defaults only -- every field is overridable from the command line, so the demo can be
# pointed at any brand and question without editing this file.
DEFAULT_PROFILE = ProfileSnapshot(
    uuid="demo-profile", name="Northwind Coffee", domain="northwindcoffee.com",
    industry="specialty coffee subscriptions",
    competitors=["bluebottle.com", "trade.coffee", "atlascoffeeclub.com"],
)
DEFAULT_QUESTION = "Do we show up when people search for coffee subscriptions?"

# fail_first_n is exact, not random: the tool fails its first N attempts and then
# succeeds. Retries are budgeted at RETRY_MAX_ATTEMPTS (4 by default), so 2 recovers
# inside the budget and 99 never can -- that is how run 3 reaches the give-up path
# instead of merely being slow.
SCENARIOS = [
    ("1. baseline - nothing broken", None),
    ("2. simulated transient failure - Google results fail twice, then recover",
     {"google_serp": 2}),
    ("3. simulated total outage - every data source down for good",
     {"google_serp": 99, "keyword_metrics": 99, "chatgpt_response": 99}),
]

TOOL_LABEL = {
    "google_serp": "Google search results",
    "keyword_metrics": "Search volume and difficulty",
    "chatgpt_response": "ChatGPT's answer",
}


def run(label: str, fail_first_n: dict[str, int] | None,
        profile: ProfileSnapshot, question: str) -> dict[str, Any]:
    settings = get_settings()
    graph = build_graph(
        llm=ScriptedToolCallingLLM(),   # pinned, whatever OPENAI_API_KEY says
        client=DataForSEOClient(settings, backend=MockBackend(
            domain_hint=profile.domain, competitors=profile.competitors,
            fail_first_n=fail_first_n)),
        settings=settings,
    )
    correlation_id = new_correlation_id()
    print(f"\n=== {label}  (correlation_id={correlation_id}) ===")
    with bind_run(correlation_id):
        state = graph.invoke(initial_state(profile=profile, question=question,
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

    # Console gets the shape of the run in words; the file carries every field.
    totals, vis = scenario["totals"], document["visibility"]
    print(f"  outcome     {document['status']}"
          f"{' (degraded)' if document['degraded'] else ''}")
    print(f"  steps       {' -> '.join(scenario['path'])}")
    print(f"  cost        {totals['api_calls']} provider calls, "
          f"{totals['retries']} retries, {totals['duration_ms']:.0f} ms")
    print(f"  visibility  {vis['visible']} visible, {vis['not_visible']} not visible, "
          f"{vis['unknown']} not measured")
    if scenario["errors"]:
        print(f"  errors      {len(scenario['errors'])} recorded "
              f"-- injected by this scenario, and expected:")
        for err in scenario["errors"]:
            print(f"                {TOOL_LABEL.get(err['tool'], err['tool'])}: "
                  f"{err['message']} "
                  f"({err['attempts']} attempts, classified {err['classification']})")
    return scenario


def simple_example(scenario: dict[str, Any], profile: ProfileSnapshot,
                   question: str) -> None:
    """One query walked end to end, before the failure scenarios add any noise.

    Everything below already happened in run 1; this only re-reads it in plain words,
    because three JSON blobs are not an introduction to anything.
    """
    report = scenario["report"]
    row = (next((q for q in report["queries"] if q["domain_visible"] is True), None)
           or next((q for q in report["queries"] if q["domain_visible"] is not None), None))
    print("\n=== A simple example: one search from run 1, start to finish ===")
    print(f'  We asked      "{question}"')
    print(f"  About         {profile.name} ({profile.domain})")
    print(f"  It planned    {len(scenario['planned_calls'])} provider calls covering "
          f"{report['queries_analysed']} search phrases")
    if row is None:
        print("  (no query returned enough data to walk through)")
        return

    evidence = row.get("evidence") or {}
    organic = evidence.get("organic") or {}
    print(f'\n  Following just one of them: "{row["query_text"]}"')
    seen = (f"{profile.domain} found at position {row['visibility_position']}"
            if row["domain_visible"] else f"{profile.domain} not in the results")
    print(f"    1. Google search results   -> {seen} "
          f"of {organic.get('results_inspected', '?')} inspected")
    print(f"    2. Search volume           -> {row['search_volume']:,} searches/month, "
          f"difficulty {row['competitive_difficulty']:.0f}/100")
    print(f"    3. ChatGPT's answer        -> {profile.name} "
          f"{'is' if row['chatgpt_mentioned'] else 'is not'} mentioned")
    print(f"    => opportunity {row['opportunity_score']} "
          f"(demand, difficulty and current visibility combined into one 0-1 number)")

    rec = next((r for r in report["recommendations"]
                if r["target_query_key"] == row["query_key"]), None)
    if rec:
        print(f"    => suggested action: {rec['title']} ({rec['content_type']})")
    print("\n  The three runs below ask that same question again, with parts of the")
    print("  system deliberately broken, to show what the answer looks like then.")


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
    parser.add_argument("--verbose", action="store_true",
                        help="also stream the structured logs to the console")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="the question to put to the pipeline")
    parser.add_argument("--brand", default=DEFAULT_PROFILE.name)
    parser.add_argument("--domain", default=DEFAULT_PROFILE.domain)
    parser.add_argument("--industry", default=DEFAULT_PROFILE.industry)
    parser.add_argument("--competitors", default=",".join(DEFAULT_PROFILE.competitors),
                        help="comma-separated")
    args = parser.parse_args()

    question = args.question
    profile = ProfileSnapshot(
        uuid="demo-profile", name=args.brand, domain=args.domain,
        industry=args.industry,
        competitors=[c.strip() for c in args.competitors.split(",") if c.strip()],
    )

    configure_logging(args.log_level, args.log_format)
    if not args.verbose:
        # 57 structured log lines on stdout bury the three paragraphs worth reading.
        # They still reach the file below, which is where you would grep them anyway.
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.CRITICAL + 1)
    if args.log_file:
        Path(args.log_file).unlink(missing_ok=True)
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(JsonFormatter())      # always JSON, whatever the console uses
        logging.getLogger().addHandler(handler)

    settings = get_settings()
    artifact: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "profile": profile.model_dump(),
        "config": {
            # Both of the demo's dependencies are pinned, so record what actually ran
            # rather than what the environment would have selected.
            "llm_mode": llm_mode(ScriptedToolCallingLLM()),
            "planner": "deterministic (pinned by the demo so all three runs plan alike)",
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
        "scenarios": [],
    }

    print("=== What this demo does ===")
    print("  Runs the whole pipeline three times against fixture data -- no network, no")
    print("  API keys, no cost. Runs 2 and 3 break parts of it on purpose; the failures")
    print("  they report are injected by the demo, not faults in the system.")

    for index, (label, fail) in enumerate(SCENARIOS):
        artifact["scenarios"].append(run(label, fail, profile, question))
        if index == 0:
            simple_example(artifact["scenarios"][0], profile, question)

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
