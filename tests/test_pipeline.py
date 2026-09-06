"""Happy path, partial success, and LLM-failure containment (PLAN §9.1, §9.4, §9.7)."""
from app.llm import ScriptedToolCallingLLM
from app.tools.mock import MockBackend

from tests.conftest import DOMAIN, QUESTION, PlannerFailsLLM


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


def test_failed_calls_say_which_query_they_cost(make_run):
    """Two failed google_serp calls are indistinguishable without query attribution."""
    state = make_run(backend=MockBackend(
        domain_hint=DOMAIN,
        fail_first_n={"google_serp": 99, "keyword_metrics": 99, "chatgpt_response": 99},
    ))
    provider_errors = [e for e in state["errors"] if e.kind == "provider"]
    assert provider_errors
    assert all(e.query_keys for e in provider_errors), "every failure names its queries"

    serp_failures = [e for e in provider_errors if e.tool == "google_serp"]
    assert len(serp_failures) == 2
    assert serp_failures[0].query_keys != serp_failures[1].query_keys, (
        "the two SERP failures must be distinguishable"
    )
    # every attributed key corresponds to a real planned query
    planned = {p.query_key for p in state["planned_queries"]}
    for error in provider_errors:
        assert set(error.query_keys) <= planned


def test_no_opportunity_ranking_when_nothing_was_measured(make_run):
    """With every dependency down all rows tie on defaults; "largest" would be noise."""
    state = make_run(backend=MockBackend(
        domain_hint=DOMAIN,
        fail_first_n={"google_serp": 99, "keyword_metrics": 99, "chatgpt_response": 99},
    ))
    document = state["report_document"]
    scores = {q["opportunity_score"] for q in document["queries"]}

    assert len(scores) == 1, "all failed rows score identically, so ranking is arbitrary"
    assert "Largest opportunity" not in document["summary"]
    assert "no opportunity ranking is meaningful" in document["summary"]


def test_opportunity_ranking_survives_a_partial_run(make_run):
    """Losing one tool must not suppress the ranking built from the others."""
    state = make_run(backend=MockBackend(domain_hint=DOMAIN,
                                         fail_first_n={"keyword_metrics": 99}))
    assert "Largest opportunity" in state["report_document"]["summary"]


def test_planner_measures_ai_overview(make_run):
    """AI Overview presence is part of the brief, and the flag defaults to off."""
    state = make_run()
    serp_calls = [c for c in state["tool_calls"] if c.name == "google_serp"]
    assert serp_calls
    assert all(c.args.get("load_async_ai_overview") for c in serp_calls)
    assert any(r.source == "ai_overview" for r in state["normalized"]), (
        "an ai_overview record should reach normalization"
    )
    assert any(q["ai_overview_mentioned"] is not None
               for q in state["report_document"]["queries"])


def test_planner_budget_is_enforced_not_merely_requested(make_run, settings):
    """The prompt asks for a budget; a model is free to ignore a prompt. Every extra call
    is billable, so the ceiling has to hold in code before retrieval (PLAN §3)."""
    greedy = [[
        *({"name": "google_serp",
           "args": {"keyword": f"query {n}", "location_code": 2840, "language_code": "en",
                    "depth": 10, "load_async_ai_overview": True},
           "id": f"s{n}", "type": "tool_call"} for n in range(6)),
        {"name": "keyword_metrics",
         "args": {"keywords": [f"query {n}" for n in range(20)]},
         "id": "k1", "type": "tool_call"},
        {"name": "chatgpt_response",
         "args": {"query_text": "something else entirely", "user_prompt": "hello",
                  "model_name": "gpt-4o-mini"},
         "id": "c1", "type": "tool_call"},
    ]]
    state = make_run(llm=ScriptedToolCallingLLM(tool_call_script=greedy))
    calls = state["tool_calls"]

    assert sum(c.name == "google_serp" for c in calls) == 2
    assert sum(c.name == "keyword_metrics" for c in calls) == 1
    assert sum(c.name == "chatgpt_response" for c in calls) == 1
    assert len({q.query_key for q in state["planned_queries"]}) <= settings.max_planned_queries
    metrics = next(c for c in calls if c.name == "keyword_metrics")
    assert len(metrics.args["keywords"]) <= settings.max_planned_queries
    # 8 planned calls asked for, 4 executed: the run cannot outspend its budget
    assert sum(e.api_calls for e in state["node_events"]) == len(calls)


def test_chatgpt_query_text_cannot_mint_its_own_query_row(make_run):
    """A planner left to itself passes the user's whole question as chatgpt query_text.
    That mints a query_key nothing else measures, so one logical query becomes two rows
    and the second carries no ranking or volume data (PLAN §6.1)."""
    script = [[
        {"name": "google_serp",
         "args": {"keyword": "agile planning tools", "location_code": 2840,
                  "language_code": "en", "depth": 10, "load_async_ai_overview": True},
         "id": "s1", "type": "tool_call"},
        {"name": "chatgpt_response",
         "args": {"query_text": QUESTION, "user_prompt": QUESTION,
                  "model_name": "gpt-4o-mini"},
         "id": "c1", "type": "tool_call"},
    ]]
    state = make_run(llm=ScriptedToolCallingLLM(tool_call_script=script))

    assert {q.query_key for q in state["planned_queries"]} == {"agile planning tools"}
    chat = next(c for c in state["tool_calls"] if c.name == "chatgpt_response")
    assert chat.args["query_text"] == "agile planning tools"
    # and the ChatGPT evidence lands on the row that also has the organic data
    row = state["merged"][0]
    assert set(row.sources) >= {"chatgpt", "organic"}
