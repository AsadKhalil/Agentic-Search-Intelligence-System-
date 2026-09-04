"""Tool-input schemas, pipeline record types, and the HTTP request/response models.

Schema discipline (PLAN §0): only *tool inputs* and *normalized outputs* are modelled.
Provider responses are consumed by defensive dict traversal against documented paths.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Query identity (PLAN §6)
# --------------------------------------------------------------------------

def query_key(text: str) -> str:
    """The stable join key: tool call -> normalized record -> query row.

    Derived once from planner-supplied text; never from provider response data.
    """
    return " ".join(text.lower().split())


# --------------------------------------------------------------------------
# Tool inputs -- the validation gate between the LLM and any paid HTTP call.
# Limits are verified against the DataForSEO docs (PLAN §14).
# extra="forbid": a hallucinated field fails locally instead of costing a call.
# --------------------------------------------------------------------------

_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def is_reasoning_model(name: str) -> bool:
    return name.lower().startswith(_REASONING_PREFIXES)


class GoogleSerpArgs(BaseModel):
    """POST /v3/serp/google/organic/live/advanced"""

    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(..., min_length=1, max_length=700,
                         description="The search query to run on Google.")
    location_code: int = Field(2840, description="DataForSEO location code; 2840 = United States.")
    language_code: str = Field("en", min_length=2, max_length=10)
    device: Literal["desktop", "mobile"] = "desktop"
    depth: int = Field(10, ge=1, le=200, description="Number of organic results to inspect.")
    load_async_ai_overview: bool = Field(
        False, description="Fetch the AI Overview block. Costs an extra $0.002 per call."
    )

    def query_texts(self) -> list[str]:
        return [self.keyword]


class KeywordMetricsArgs(BaseModel):
    """POST /v3/dataforseo_labs/google/keyword_overview/live"""

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(..., min_length=1, max_length=700,
                                description="Keywords to fetch search volume and difficulty for.")
    location_code: int = Field(2840)
    language_code: str = Field("en", min_length=2, max_length=10)

    @field_validator("keywords")
    @classmethod
    def _per_keyword_limits(cls, keywords: list[str]) -> list[str]:
        for kw in keywords:
            if not kw.strip():
                raise ValueError("keyword must not be blank")
            if len(kw) > 80:
                raise ValueError(f"keyword exceeds 80 characters: {kw[:40]!r}...")
            if len(kw.split()) > 10:
                raise ValueError(f"keyword exceeds 10 words: {kw!r}")
        return keywords

    def query_texts(self) -> list[str]:
        return list(self.keywords)


class ChatGptResponseArgs(BaseModel):
    """POST /v3/ai_optimization/chat_gpt/llm_responses/live"""

    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(..., min_length=1, max_length=500,
                            description="The logical search query this prompt investigates.")
    user_prompt: str = Field(..., min_length=1, max_length=500,
                             description="Prompt sent to ChatGPT. Hard limit 500 characters.")
    model_name: str = Field(..., description="e.g. gpt-4o-mini")
    system_message: str | None = Field(None, max_length=500)
    max_output_tokens: int = Field(2048, ge=16, le=4096)
    temperature: float | None = Field(None, ge=0, le=2)
    web_search: bool = False
    message_chain: list[dict[str, Any]] | None = Field(None, max_length=10)

    @model_validator(mode="after")
    def _reasoning_token_floor(self) -> "ChatGptResponseArgs":
        if is_reasoning_model(self.model_name) and self.max_output_tokens < 1024:
            raise ValueError(
                f"reasoning model {self.model_name!r} requires max_output_tokens >= 1024"
            )
        return self

    def query_texts(self) -> list[str]:
        return [self.query_text]


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "google_serp": GoogleSerpArgs,
    "keyword_metrics": KeywordMetricsArgs,
    "chatgpt_response": ChatGptResponseArgs,
}


# --------------------------------------------------------------------------
# Pipeline records
# --------------------------------------------------------------------------

Source = Literal["organic", "ai_overview", "keyword_metrics", "chatgpt"]
RetrievalStatus = Literal["ok", "partial", "failed"]


class ProfileSnapshot(BaseModel):
    """The bits of a profile the graph needs; keeps nodes free of ORM objects."""

    uuid: str
    name: str
    domain: str
    industry: str | None = None
    description: str | None = None
    competitors: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """Mirror of langchain's AIMessage.tool_call dict, kept as a typed record in state."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class PlannedQuery(BaseModel):
    """One distinct logical query the plan covers. Its key is assigned at plan time and
    is the join key all the way through to the queries table (PLAN §6.1)."""

    query_key: str
    query_text: str
    tools: list[str] = Field(default_factory=list)
    failed_tools: list[str] = Field(default_factory=list)


class RawPayload(BaseModel):
    tool: str
    query_texts: list[str]
    payload: dict[str, Any]
    classification: str
    usable: bool


class PipelineError(BaseModel):
    node: str
    kind: str                      # tool_argument | provider | transport | llm | schema
    tool: str | None = None
    classification: str | None = None
    status_code: int | None = None
    message: str
    attempts: int = 1


class NodeEvent(BaseModel):
    node: str
    ok: bool
    duration_ms: float
    retries: int = 0
    api_calls: int = 0
    tokens: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class NormalizedRecord(BaseModel):
    """One provider surface's view of one logical query."""

    query_key: str
    query_text: str
    source: Source
    search_volume: int | None = None
    competitive_difficulty: float | None = None
    domain_visible: bool | None = None          # organic only (PLAN §6.4)
    visibility_position: int | None = None
    ai_overview_mentioned: bool | None = None
    chatgpt_mentioned: bool | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class MergedQuery(BaseModel):
    """The per-query row the analyzer scores and persistence writes."""

    query_key: str
    query_text: str
    retrieval_status: RetrievalStatus = "failed"
    sources: list[Source] = Field(default_factory=list)
    search_volume: int | None = None
    competitive_difficulty: float | None = None
    domain_visible: bool | None = None
    visibility_position: int | None = None
    ai_overview_mentioned: bool | None = None
    chatgpt_mentioned: bool | None = None
    opportunity_score: float = 0.0
    evidence: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Analyzer output -- Pydantic-validated; a violation routes to fallback_analysis.
# --------------------------------------------------------------------------

class LLMInsight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query_key: str
    rationale: str = Field(..., min_length=1, max_length=600)


class LLMRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target_query_key: str
    content_type: str = Field(..., min_length=1, max_length=80)
    title: str = Field(..., min_length=1, max_length=200)
    rationale: str = Field(..., min_length=1, max_length=600)
    target_keywords: list[str] = Field(default_factory=list, max_length=20)
    priority: Literal["high", "medium", "low"] = "medium"


class LLMAnalysis(BaseModel):
    """What the analyzer LLM is allowed to decide. Scores stay deterministic (PLAN §7)."""

    model_config = ConfigDict(extra="ignore")
    summary: str = Field(..., min_length=1, max_length=2000)
    insights: list[LLMInsight] = Field(default_factory=list, max_length=50)
    recommendations: list[LLMRecommendation] = Field(default_factory=list, max_length=50)


class Insight(BaseModel):
    query_key: str
    query_text: str
    opportunity_score: float
    domain_visible: bool | None = None
    visibility_position: int | None = None
    search_volume: int | None = None
    competitive_difficulty: float | None = None
    rationale: str


class Recommendation(BaseModel):
    target_query_key: str
    content_type: str
    title: str
    rationale: str
    target_keywords: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"


class AnalysisResult(BaseModel):
    summary: str
    insights: list[Insight] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    generated_by: Literal["llm", "deterministic"] = "llm"


# --------------------------------------------------------------------------
# HTTP models
# --------------------------------------------------------------------------

class ProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=200)
    domain: str = Field(..., min_length=3, max_length=253)
    industry: str | None = Field(None, max_length=120)
    description: str | None = Field(None, max_length=2000)
    competitors: list[str] = Field(default_factory=list, max_length=25)


class ProfileOut(BaseModel):
    uuid: str
    name: str
    domain: str
    industry: str | None
    description: str | None
    competitors: list[str]
    created_at: datetime
    total_runs: int = 0
    last_run_status: str | None = None
    average_opportunity_score: float | None = None


class RunRequest(BaseModel):
    """POST /profiles/{uuid}/run -- the question is required; 422 without it."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=3, max_length=1000)


class QueryOut(BaseModel):
    uuid: str
    run_uuid: str
    query_text: str
    query_key: str
    retrieval_status: str
    estimated_search_volume: int | None
    competitive_difficulty: float | None
    domain_visible: bool | None
    visibility_position: int | None
    ai_overview_mentioned: bool | None
    chatgpt_mentioned: bool | None
    opportunity_score: float
    discovered_at: datetime


class RecommendationOut(BaseModel):
    uuid: str
    run_uuid: str
    target_query_uuid: str
    content_type: str
    title: str
    rationale: str
    target_keywords: list[str]
    priority: str


class Page(BaseModel):
    items: list[Any]
    page: int
    per_page: int
    total: int


class RunResponse(BaseModel):
    run_uuid: str
    profile_uuid: str
    kind: str
    status: str
    degraded: bool
    correlation_id: str
    question: str
    planned_call_count: int
    extracted_record_count: int
    tokens_used: int
    errors: list[PipelineError]
    insights: list[Insight]
    recommendations: list[Recommendation]
    report: dict[str, Any]
    metrics: dict[str, Any]
