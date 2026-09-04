"""Graph state.

LangGraph *overwrites* ordinary keys and only merges reducer-annotated ones, so anything
several nodes contribute to must carry an explicit reducer.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.schemas import (
    AnalysisResult,
    MergedQuery,
    NodeEvent,
    NormalizedRecord,
    PipelineError,
    PlannedQuery,
    ProfileSnapshot,
    RawPayload,
    ToolCall,
)


class PipelineState(TypedDict, total=False):
    # --- overwritten ------------------------------------------------------
    profile: ProfileSnapshot
    question: str
    correlation_id: str
    tool_calls: list[ToolCall]
    planned_queries: list[PlannedQuery]
    normalized: list[NormalizedRecord]
    merged: list[MergedQuery]
    analysis: AnalysisResult | None
    report_document: dict[str, Any] | None
    degraded: bool
    serp_attempted: bool

    # --- accumulated ------------------------------------------------------
    raw_payloads: Annotated[list[RawPayload], operator.add]
    errors: Annotated[list[PipelineError], operator.add]
    node_events: Annotated[list[NodeEvent], operator.add]


def initial_state(
    *, profile: ProfileSnapshot, question: str, correlation_id: str
) -> PipelineState:
    return PipelineState(
        profile=profile,
        question=question,
        correlation_id=correlation_id,
        tool_calls=[],
        planned_queries=[],
        normalized=[],
        merged=[],
        analysis=None,
        report_document=None,
        degraded=False,
        serp_attempted=False,
        raw_payloads=[],
        errors=[],
        node_events=[],
    )
