"""Happy path, partial success, and LLM-failure containment (PLAN §9.1, §9.4, §9.7)."""
from app.llm import ScriptedToolCallingLLM
from app.tools.mock import MockBackend

from tests.conftest import DOMAIN, PlannerFailsLLM


def test_happy_path_completes(make_run):
    state = make_run()
    document = state["report_document"]

    assert document["status"] == "completed"
    assert state["errors"] == []
    assert document["analysis_generated_by"] == "llm"
    assert [e.node for e in state["node_events"]] == [
        "plan_queries", "retrieve", "normalize", "analyze", "report",
    ]
    assert document["queries_analysed"] >= 3
    assert document["insights"] and document["recommendations"]
    # every recommendation points at a query the run actually analysed
    keys = {q["query_key"] for q in document["queries"]}
    assert all(r["target_query_key"] in keys for r in document["recommendations"])


def test_partial_success_keeps_what_was_retrieved(make_run):
    """One of three tools fails permanently; the run degrades instead of failing."""
    state = make_run(backend=MockBackend(domain_hint=DOMAIN,
                                         fail_first_n={"keyword_metrics": 99}))
    document = state["report_document"]

    assert document["status"] == "partial"
    assert state["degraded"] is True
    assert state["normalized"], "SERP and ChatGPT results should still be present"
    assert any(e.tool == "keyword_metrics" for e in state["errors"])
    assert {r.tool for r in state["raw_payloads"]} == {"google_serp", "chatgpt_response"}
    # volume is unknown for every query, but visibility was still established
    assert all(q["search_volume"] is None for q in document["queries"])
    assert any(q["domain_visible"] is not None for q in document["queries"])


def test_planner_failure_routes_to_fallback_plan(make_run):
    state = make_run(llm=PlannerFailsLLM())
    document = state["report_document"]

    nodes = [e.node for e in state["node_events"]]
    assert nodes[:3] == ["plan_queries", "fallback_plan", "retrieve"]
    assert "report" in nodes
    assert any(e.kind == "llm" and e.node == "plan_queries" for e in state["errors"])
    assert document["queries_analysed"] > 0
    assert document["degraded"] is True


def test_invalid_analysis_routes_to_fallback_analysis(make_run):
    """Schema-violating analyzer output must not reach report."""
    llm = ScriptedToolCallingLLM(content_script=["I am not JSON."])
    state = make_run(llm=llm)
    document = state["report_document"]

    nodes = [e.node for e in state["node_events"]]
    assert nodes[-3:] == ["analyze", "fallback_analysis", "report"]
    assert document["analysis_generated_by"] == "deterministic"
    assert any(e.kind == "schema" for e in state["errors"])
    # report still received a valid, fully populated analysis
    assert state["analysis"] is not None
    assert document["insights"] and document["summary"]


def test_fallback_serp_fires_only_when_serp_never_attempted(make_run):
    """A plan with no SERP call gets one; an already-failed SERP is not retried."""
    no_serp = [[{"name": "keyword_metrics",
                 "args": {"keywords": ["best project management tools"]},
                 "id": "c1", "type": "tool_call"}]]
    state = make_run(llm=ScriptedToolCallingLLM(tool_call_script=no_serp))
    assert "fallback_serp" not in [e.node for e in state["node_events"]], (
        "keyword_metrics succeeded, so no fallback is needed"
    )

    state = make_run(
        llm=ScriptedToolCallingLLM(tool_call_script=[list(no_serp[0])]),
        backend=MockBackend(domain_hint=DOMAIN, fail_first_n={"keyword_metrics": 99}),
    )
    nodes = [e.node for e in state["node_events"]]
    assert "fallback_serp" in nodes
    assert state["serp_attempted"] is True
    assert state["normalized"], "the fallback SERP call recovered usable data"


def test_serp_already_failed_is_not_retried_by_fallback(make_run):
    state = make_run(backend=MockBackend(
        domain_hint=DOMAIN,
        fail_first_n={"google_serp": 99, "keyword_metrics": 99, "chatgpt_response": 99},
    ))
    nodes = [e.node for e in state["node_events"]]
    assert "fallback_serp" not in nodes
    assert state["report_document"]["status"] == "partial"
