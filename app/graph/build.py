"""Graph wiring: the main pipeline and the recheck subgraph.

One refinement on the planned routing table: when retrieval yields nothing usable the
graph still passes through `normalize` rather than jumping straight to `report`. Skipping
it would leave `merged` empty, and then no queries row would be written for the failed
queries -- which is exactly the row PLAN §6.3 requires and the one most worth rechecking.
The outcome (status `partial`, no LLM analysis) is unchanged; only the path is.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.graph import nodes
from app.graph.nodes import Deps
from app.graph.state import PipelineState
from app.llm import get_llm
from app.tools.dataforseo import DataForSEOClient

# --------------------------------------------------------------------------
# Routers
# --------------------------------------------------------------------------

def route_after_plan(state: PipelineState) -> str:
    return "retrieve" if state.get("tool_calls") else "fallback_plan"


def route_after_retrieval(state: PipelineState) -> str:
    if any(p.usable for p in state.get("raw_payloads", [])):
        return "normalize"
    # A fallback only makes sense for a dependency that was never tried; retrying an
    # already-failed one is theatre.
    if not state.get("serp_attempted"):
        return "fallback_serp"
    return "normalize"


def route_after_normalize(state: PipelineState) -> str:
    return "analyze" if state.get("normalized") else "fallback_analysis"


def route_after_analyze(state: PipelineState) -> str:
    return "report" if state.get("analysis") is not None else "fallback_analysis"


def _deps(llm: Any = None, client: DataForSEOClient | None = None,
          settings: Settings | None = None) -> Deps:
    s = settings or get_settings()
    return Deps(llm=llm or get_llm(s), client=client or DataForSEOClient(s), settings=s)


def _bind(deps: Deps) -> dict[str, Any]:
    return {
        name: partial(getattr(nodes, name), deps=deps)
        for name in ("plan_queries", "fallback_plan", "retrieve", "fallback_serp",
                     "normalize", "analyze", "fallback_analysis", "report")
    }


def build_graph(llm: Any = None, client: DataForSEOClient | None = None,
                settings: Settings | None = None):
    deps = _deps(llm, client, settings)
    fn = _bind(deps)
    g = StateGraph(PipelineState)
    for name, func in fn.items():
        g.add_node(name, func)

    g.add_edge(START, "plan_queries")
    g.add_conditional_edges("plan_queries", route_after_plan,
                            {"retrieve": "retrieve", "fallback_plan": "fallback_plan"})
    g.add_edge("fallback_plan", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieval,
                            {"normalize": "normalize", "fallback_serp": "fallback_serp"})
    g.add_edge("fallback_serp", "normalize")
    g.add_conditional_edges("normalize", route_after_normalize,
                            {"analyze": "analyze", "fallback_analysis": "fallback_analysis"})
    g.add_conditional_edges("analyze", route_after_analyze,
                            {"report": "report", "fallback_analysis": "fallback_analysis"})
    g.add_edge("fallback_analysis", "report")
    g.add_edge("report", END)
    return g.compile()


def build_recheck_graph(llm: Any = None, client: DataForSEOClient | None = None,
                        settings: Settings | None = None):
    """Recheck reconstructs one query's calls deterministically -- no planner LLM."""
    deps = _deps(llm, client, settings)
    fn = _bind(deps)
    g = StateGraph(PipelineState)
    for name in ("retrieve", "fallback_serp", "normalize", "analyze",
                 "fallback_analysis", "report"):
        g.add_node(name, fn[name])

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieval,
                            {"normalize": "normalize", "fallback_serp": "fallback_serp"})
    g.add_edge("fallback_serp", "normalize")
    g.add_conditional_edges("normalize", route_after_normalize,
                            {"analyze": "analyze", "fallback_analysis": "fallback_analysis"})
    g.add_conditional_edges("analyze", route_after_analyze,
                            {"report": "report", "fallback_analysis": "fallback_analysis"})
    g.add_edge("fallback_analysis", "report")
    g.add_edge("report", END)
    return g.compile()
