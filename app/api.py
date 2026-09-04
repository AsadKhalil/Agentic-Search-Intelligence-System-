"""HTTP surface and persistence (PLAN §6, §8).

Synchronous by design: the brief accepts it, and a BackgroundTask would still not survive
a restart, so it would buy complexity without buying durability.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_session
from app.graph.build import build_graph, build_recheck_graph
from app.graph.state import initial_state
from app.llm import get_llm, single_query_tool_calls
from app.models import PipelineRun, Profile, Query, Recommendation, new_uuid, utcnow
from app.observability.logging import bind_run, get_logger, new_correlation_id
from app.observability.metrics import RunMetrics
from app.schemas import (
    Page,
    PlannedQuery,
    ProfileCreate,
    ProfileOut,
    ProfileSnapshot,
    QueryOut,
    RecommendationOut,
    RunRequest,
    RunResponse,
    ToolCall,
    query_key,
)
from app.tools.dataforseo import DataForSEOClient
from app.tools.mock import MockBackend

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1")


class Runner:
    """Builds the pipeline for one profile.

    In mock mode the backend is given the profile's domain as a hint so the offline demo
    produces both visible and not-visible queries; a real SERP request carries no domain.
    """

    def __init__(self, llm: Any = None, client: DataForSEOClient | None = None,
                 settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self._client = client

    def client_for(self, profile: ProfileSnapshot) -> DataForSEOClient:
        if self._client is not None:
            return self._client
        if self.settings.mock_dataforseo:
            return DataForSEOClient(self.settings, backend=MockBackend(
                domain_hint=profile.domain, latency_ms=self.settings.mock_latency_ms))
        return DataForSEOClient(self.settings)

    def graph_for(self, profile: ProfileSnapshot):
        return build_graph(llm=self._llm or get_llm(self.settings),
                           client=self.client_for(profile), settings=self.settings)

    def recheck_graph_for(self, profile: ProfileSnapshot):
        return build_recheck_graph(llm=self._llm or get_llm(self.settings),
                                   client=self.client_for(profile), settings=self.settings)


@lru_cache
def get_runner() -> Runner:
    return Runner()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _snapshot(profile: Profile) -> ProfileSnapshot:
    return ProfileSnapshot(
        uuid=profile.uuid, name=profile.name, domain=profile.domain,
        industry=profile.industry, description=profile.description,
        competitors=list(profile.competitors or []),
    )


def _load_profile(session: Session, profile_uuid: str) -> Profile:
    profile = session.get(Profile, profile_uuid)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    return profile


def _latest_full_run_uuid(session: Session, profile_uuid: str) -> str | None:
    return session.scalar(
        select(PipelineRun.uuid)
        .where(PipelineRun.profile_uuid == profile_uuid, PipelineRun.kind == "full")
        .order_by(PipelineRun.started_at.desc())
        .limit(1)
    )


def _persist(session: Session, profile: Profile, state: dict[str, Any], *, kind: str,
             question: str, started: datetime, duration_ms: float) -> PipelineRun:
    document = state.get("report_document") or {}
    metrics = RunMetrics(state.get("node_events", []))
    merged = state.get("merged", [])
    analysis = state.get("analysis")

    run = PipelineRun(
        uuid=new_uuid(), profile_uuid=profile.uuid, kind=kind,
        status=document.get("status", "partial"), question=question,
        correlation_id=state.get("correlation_id", ""),
        degraded=bool(state.get("degraded")),
        planned_call_count=len(state.get("tool_calls", [])),
        extracted_record_count=len(state.get("normalized", [])),
        tokens_used=metrics.tokens_used, started_at=started, finished_at=utcnow(),
        duration_ms=round(duration_ms, 3),
        errors=[e.model_dump() for e in state.get("errors", [])],
        report=document, metrics=metrics.as_dict(),
    )
    session.add(run)

    key_to_uuid: dict[str, str] = {}
    for row in merged:
        record = Query(
            uuid=new_uuid(), run_uuid=run.uuid, profile_uuid=profile.uuid,
            query_text=row.query_text, query_key=row.query_key,
            retrieval_status=row.retrieval_status,
            estimated_search_volume=row.search_volume,
            competitive_difficulty=row.competitive_difficulty,
            domain_visible=row.domain_visible,
            visibility_position=row.visibility_position,
            ai_overview_mentioned=row.ai_overview_mentioned,
            chatgpt_mentioned=row.chatgpt_mentioned,
            opportunity_score=row.opportunity_score, evidence=row.evidence,
        )
        session.add(record)
        key_to_uuid[row.query_key] = record.uuid

    for rec in (analysis.recommendations if analysis else []):
        target = key_to_uuid.get(rec.target_query_key)
        if target is None:
            continue
        session.add(Recommendation(
            uuid=new_uuid(), run_uuid=run.uuid, target_query_uuid=target,
            content_type=rec.content_type, title=rec.title, rationale=rec.rationale,
            target_keywords=rec.target_keywords, priority=rec.priority,
        ))
    session.commit()
    return run


def _run_response(run: PipelineRun, state: dict[str, Any]) -> RunResponse:
    analysis = state.get("analysis")
    return RunResponse(
        run_uuid=run.uuid, profile_uuid=run.profile_uuid, kind=run.kind,
        status=run.status, degraded=run.degraded,
        correlation_id=run.correlation_id, question=run.question,
        planned_call_count=run.planned_call_count,
        extracted_record_count=run.extracted_record_count,
        tokens_used=run.tokens_used, errors=state.get("errors", []),
        insights=analysis.insights if analysis else [],
        recommendations=analysis.recommendations if analysis else [],
        report=run.report, metrics=run.metrics,
    )


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------

@router.post("/profiles", response_model=ProfileOut, status_code=201)
def create_profile(body: ProfileCreate, session: Session = Depends(get_session)) -> ProfileOut:
    profile = Profile(uuid=new_uuid(), **body.model_dump())
    session.add(profile)
    session.commit()
    return ProfileOut(**body.model_dump(), uuid=profile.uuid, created_at=profile.created_at)


@router.get("/profiles/{profile_uuid}", response_model=ProfileOut)
def read_profile(profile_uuid: str, session: Session = Depends(get_session)) -> ProfileOut:
    profile = _load_profile(session, profile_uuid)
    total_runs = session.scalar(
        select(func.count()).select_from(PipelineRun)
        .where(PipelineRun.profile_uuid == profile_uuid)
    ) or 0
    last_status = session.scalar(
        select(PipelineRun.status).where(PipelineRun.profile_uuid == profile_uuid)
        .order_by(PipelineRun.started_at.desc()).limit(1)
    )
    latest = _latest_full_run_uuid(session, profile_uuid)
    average = session.scalar(
        select(func.avg(Query.opportunity_score)).where(Query.run_uuid == latest)
    ) if latest else None
    return ProfileOut(
        uuid=profile.uuid, name=profile.name, domain=profile.domain,
        industry=profile.industry, description=profile.description,
        competitors=list(profile.competitors or []), created_at=profile.created_at,
        total_runs=total_runs, last_run_status=last_status,
        average_opportunity_score=round(average, 4) if average is not None else None,
    )


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

@router.post("/profiles/{profile_uuid}/run", response_model=RunResponse)
def run_pipeline(profile_uuid: str, body: RunRequest,
                 session: Session = Depends(get_session),
                 runner: Runner = Depends(get_runner)) -> RunResponse:
    profile = _load_profile(session, profile_uuid)
    snapshot = _snapshot(profile)
    correlation_id = new_correlation_id()
    started, clock = utcnow(), perf_counter()

    with bind_run(correlation_id):
        log.info("run.start", extra={"profile_uuid": profile_uuid, "kind": "full"})
        state = runner.graph_for(snapshot).invoke(
            initial_state(profile=snapshot, question=body.question,
                          correlation_id=correlation_id)
        )
        run = _persist(session, profile, state, kind="full", question=body.question,
                       started=started, duration_ms=(perf_counter() - clock) * 1000)
        log.info("run.finish", extra={"run_uuid": run.uuid, "status": run.status,
                                      "degraded": run.degraded})
    return _run_response(run, state)


@router.post("/queries/{query_uuid}/recheck", response_model=RunResponse)
def recheck_query(query_uuid: str, session: Session = Depends(get_session),
                  runner: Runner = Depends(get_runner),
                  settings: Settings = Depends(get_settings)) -> RunResponse:
    record = session.get(Query, query_uuid)
    if record is None:
        raise HTTPException(status_code=404, detail="query not found")
    profile = _load_profile(session, record.profile_uuid)
    snapshot = _snapshot(profile)
    correlation_id = new_correlation_id()
    started, clock = utcnow(), perf_counter()

    calls = single_query_tool_calls(record.query_text, model_name=settings.llm_model)
    state_in = initial_state(profile=snapshot, question=f"Recheck: {record.query_text}",
                             correlation_id=correlation_id)
    state_in["tool_calls"] = [ToolCall(name=c["name"], args=c["args"], id=c["id"])
                              for c in calls]
    state_in["planned_queries"] = [PlannedQuery(query_key=query_key(record.query_text),
                                                query_text=record.query_text)]

    with bind_run(correlation_id):
        log.info("run.start", extra={"profile_uuid": profile.uuid, "kind": "recheck",
                                      "query_uuid": query_uuid})
        state = runner.recheck_graph_for(snapshot).invoke(state_in)
        run = _persist(session, profile, state, kind="recheck",
                       question=f"Recheck: {record.query_text}", started=started,
                       duration_ms=(perf_counter() - clock) * 1000)

        # Update the existing query row in place and replace its recommendations, so the
        # profile's latest-full-run view reflects the recheck (PLAN §6).
        fresh = next((m for m in state.get("merged", [])
                      if m.query_key == record.query_key), None)
        if fresh is not None:
            record.retrieval_status = fresh.retrieval_status
            record.estimated_search_volume = fresh.search_volume
            record.competitive_difficulty = fresh.competitive_difficulty
            record.domain_visible = fresh.domain_visible
            record.visibility_position = fresh.visibility_position
            record.ai_overview_mentioned = fresh.ai_overview_mentioned
            record.chatgpt_mentioned = fresh.chatgpt_mentioned
            record.opportunity_score = fresh.opportunity_score
            record.evidence = fresh.evidence
            record.updated_at = utcnow()

        session.query(Recommendation).filter(
            Recommendation.target_query_uuid == record.uuid
        ).delete(synchronize_session=False)
        analysis = state.get("analysis")
        for rec in (analysis.recommendations if analysis else []):
            if rec.target_query_key != record.query_key:
                continue
            session.add(Recommendation(
                uuid=new_uuid(), run_uuid=run.uuid, target_query_uuid=record.uuid,
                content_type=rec.content_type, title=rec.title, rationale=rec.rationale,
                target_keywords=rec.target_keywords, priority=rec.priority,
            ))
        # The recheck run's own duplicate query rows are noise; the canonical row is the
        # one just updated in place.
        session.query(Query).filter(Query.run_uuid == run.uuid).delete(
            synchronize_session=False)
        session.commit()
        log.info("run.finish", extra={"run_uuid": run.uuid, "status": run.status})
    return _run_response(run, state)


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------

@router.get("/profiles/{profile_uuid}/queries", response_model=Page)
def list_queries(
    profile_uuid: str,
    min_score: float | None = QueryParam(None, ge=0, le=1),
    status: str | None = QueryParam(None, pattern="^(visible|not_visible|unknown)$"),
    page: int = QueryParam(1, ge=1),
    per_page: int = QueryParam(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> Page:
    _load_profile(session, profile_uuid)
    latest = _latest_full_run_uuid(session, profile_uuid)
    if latest is None:
        return Page(items=[], page=page, per_page=per_page, total=0)

    stmt = select(Query).where(Query.run_uuid == latest)
    if min_score is not None:
        stmt = stmt.where(Query.opportunity_score >= min_score)
    if status == "visible":
        stmt = stmt.where(Query.domain_visible.is_(True))
    elif status == "not_visible":
        stmt = stmt.where(Query.domain_visible.is_(False))
    elif status == "unknown":
        stmt = stmt.where(Query.domain_visible.is_(None))

    total = session.scalar(
        select(func.count()).select_from(stmt.subquery())) or 0
    rows = session.scalars(
        stmt.order_by(Query.opportunity_score.desc(), Query.query_key)
        .offset((page - 1) * per_page).limit(per_page)
    ).all()
    return Page(items=[QueryOut.model_validate(r, from_attributes=True) for r in rows],
                page=page, per_page=per_page, total=total)


@router.get("/profiles/{profile_uuid}/recommendations", response_model=Page)
def list_recommendations(
    profile_uuid: str,
    page: int = QueryParam(1, ge=1),
    per_page: int = QueryParam(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> Page:
    _load_profile(session, profile_uuid)
    latest = _latest_full_run_uuid(session, profile_uuid)
    if latest is None:
        return Page(items=[], page=page, per_page=per_page, total=0)

    # Joined through the latest full run's query rows, not by recommendation.run_uuid:
    # a rechecked recommendation is written under the recheck run and would vanish.
    stmt = (select(Recommendation).join(Query, Query.uuid == Recommendation.target_query_uuid)
            .where(Query.run_uuid == latest))
    total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = session.scalars(
        stmt.order_by(Query.opportunity_score.desc(), Recommendation.title)
        .offset((page - 1) * per_page).limit(per_page)
    ).all()
    return Page(items=[RecommendationOut.model_validate(r, from_attributes=True)
                       for r in rows], page=page, per_page=per_page, total=total)
