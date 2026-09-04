"""ORM tables (PLAN §6).

profiles -- pipeline_runs --+-- queries -- recommendations
                            +-- (report + metrics as JSON on the run row)
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_uuid() -> str:
    return str(_uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Profile(Base):
    __tablename__ = "profiles"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(200))
    domain: Mapped[str] = mapped_column(String(253), index=True)
    industry: Mapped[str | None] = mapped_column(String(120), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    competitors: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    runs: Mapped[list["PipelineRun"]] = relationship(back_populates="profile")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    profile_uuid: Mapped[str] = mapped_column(ForeignKey("profiles.uuid"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="full")        # full | recheck
    status: Mapped[str] = mapped_column(String(16), default="completed")  # completed|partial|failed
    question: Mapped[str] = mapped_column(Text)
    correlation_id: Mapped[str] = mapped_column(String(64), index=True)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    planned_call_count: Mapped[int] = mapped_column(Integer, default=0)
    extracted_record_count: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    errors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    profile: Mapped[Profile] = relationship(back_populates="runs")


class Query(Base):
    """One row per distinct logical query -- created even when retrieval failed, so a
    failed query stays visible and recheckable (PLAN §6.3)."""

    __tablename__ = "queries"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    run_uuid: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.uuid"), index=True)
    profile_uuid: Mapped[str] = mapped_column(ForeignKey("profiles.uuid"), index=True)
    query_text: Mapped[str] = mapped_column(Text)
    query_key: Mapped[str] = mapped_column(String(255), index=True)
    retrieval_status: Mapped[str] = mapped_column(String(16), default="failed")
    estimated_search_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    competitive_difficulty: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain_visible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    visibility_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_overview_mentioned: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    chatgpt_mentioned: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    opportunity_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="query", cascade="all, delete-orphan"
    )


class Recommendation(Base):
    __tablename__ = "recommendations"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    run_uuid: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.uuid"), index=True)
    target_query_uuid: Mapped[str] = mapped_column(ForeignKey("queries.uuid"), index=True)
    content_type: Mapped[str] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(String(200))
    rationale: Mapped[str] = mapped_column(Text)
    target_keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    query: Mapped[Query] = relationship(back_populates="recommendations")
