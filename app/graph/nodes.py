"""The eight nodes. Each does exactly one thing (PLAN §1).

Synthesis lives only in `analyze`; `report` is pure formatting, so the terminal node
cannot fail on a model error.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from time import perf_counter
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.graph.state import PipelineState
from app.llm import (
    content_type_for,
    core_phrase,
    llm_mode,
    template_tool_calls,
    tokens_from,
)
from app.observability.logging import bind_node, get_logger
from app.resilience import ProviderError, ToolArgumentError
from app.schemas import (
    AnalysisResult,
    Insight,
    LLMAnalysis,
    MergedQuery,
    NodeEvent,
    NormalizedRecord,
    PipelineError,
    PlannedQuery,
    RawPayload,
    Recommendation,
    ToolCall,
    query_key,
)
from app.scoring import apply_scores
from app.tools.dataforseo import TOOLS, DataForSEOClient, validate_args

log = get_logger(__name__)


@dataclass
class Deps:
    llm: Any
    client: DataForSEOClient
    settings: Settings


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

_META = ("_ok", "_retries", "_api_calls", "_tokens", "_detail")


def node(name: str) -> Callable:
    """Times the node, binds it to the log context, and emits a NodeEvent.

    Deliberately does not swallow exceptions: expected external failures are caught
    inside each node and routed; anything reaching here is a bug and should surface.
    """

    def decorate(fn: Callable[[PipelineState, Deps], dict[str, Any]]) -> Callable:
        @wraps(fn)
        def wrapper(state: PipelineState, deps: Deps) -> dict[str, Any]:
            with bind_node(name):
                log.info("node.start")
                started = perf_counter()
                update = fn(state, deps) or {}
                duration = (perf_counter() - started) * 1000
                event = NodeEvent(
                    node=name,
                    ok=update.pop("_ok", True),
                    duration_ms=round(duration, 3),
                    retries=update.pop("_retries", 0),
                    api_calls=update.pop("_api_calls", 0),
                    tokens=update.pop("_tokens", 0),
                    detail=update.pop("_detail", {}),
                )
                log.info(
                    "node.finish",
                    extra={"duration_ms": event.duration_ms, "ok": event.ok,
                           "retries": event.retries, "api_calls": event.api_calls,
                           **event.detail},
                )
                return {**update, "node_events": [event]}

        return wrapper

    return decorate


# --------------------------------------------------------------------------
# 1. plan_queries -- the only node that decides which tools to call
# --------------------------------------------------------------------------

# Per-tool call ceilings. The prompt below states them and `enforce_limits` applies them:
# a prompt is a request, and every extra call is a billable provider request (PLAN §3).
TOOL_CALL_LIMITS = {"google_serp": 2, "keyword_metrics": 1, "chatgpt_response": 1}

PLANNER_PROMPT = """You are the retrieval planner for a search-visibility research system.

BRAND: {name}
DOMAIN: {domain}
INDUSTRY: {industry}
COMPETITORS: {competitors}
QUESTION: {question}

Plan the DataForSEO calls needed to answer the question. Rules:
- Call google_serp for each distinct search query whose organic ranking matters
  (at most {serp_limit} calls).
- Call keyword_metrics ONCE, passing every query you are investigating.
- Set load_async_ai_overview to true on google_serp calls: whether the brand appears in
  the AI Overview is part of what we are measuring, and it is off by default.
- Call chatgpt_response ONCE. Its query_text must be one of the queries you passed to
  google_serp or keyword_metrics, verbatim -- not the user's question.
- Investigate at most {max_queries} distinct queries.
- Keywords must be at most 80 characters and 10 words.
Respond with tool calls only."""


def enforce_limits(calls: list[ToolCall], max_queries: int) -> tuple[list[ToolCall], list[str]]:
    """Trim planner output to the documented budget, before anything becomes billable.

    Three ceilings, each of which a model is free to talk itself past:

    * per-tool call counts (``TOOL_CALL_LIMITS``);
    * distinct logical queries (``settings.max_planned_queries``);
    * ``chatgpt_response.query_text`` must name a query already being measured. Left
      alone, a model tends to pass the user's whole question there, which mints a
      ``query_key`` nothing else covers and splits one logical query into two rows with
      no SERP or volume data on the second (PLAN §6.1).

    Returns the calls to execute and a human-readable note per adjustment.
    """
    kept: list[ToolCall] = []
    dropped: list[str] = []
    per_tool: dict[str, int] = {}
    budget: dict[str, str] = {}          # query_key -> the text that claimed the slot

    def admit(text: str) -> bool:
        key = query_key(text)
        if key in budget:
            return True
        if len(budget) >= max_queries:
            return False
        budget[key] = text
        return True

    # chatgpt_response is processed last so the queries it must reference are already in.
    for call in sorted(calls, key=lambda c: c.name == "chatgpt_response"):
        cap = TOOL_CALL_LIMITS.get(call.name)
        if cap is None:
            dropped.append(f"{call.name}: not a tool this planner may call")
            continue
        if per_tool.get(call.name, 0) >= cap:
            dropped.append(f"{call.name}: over its limit of {cap} call(s)")
            continue
        try:
            args = validate_args(call.name, call.args)
        except ToolArgumentError:
            kept.append(call)            # retrieve rejects it and records the real reason
            per_tool[call.name] = per_tool.get(call.name, 0) + 1
            continue

        if call.name == "chatgpt_response":
            if query_key(args.query_text) not in budget:
                if not budget:
                    dropped.append("chatgpt_response: no measured query to attach it to")
                    continue
                anchor = next(iter(budget.values()))
                dropped.append(f"chatgpt_response: query_text re-anchored to {anchor!r}")
                call = ToolCall(name=call.name, id=call.id,
                                args={**call.args, "query_text": anchor})
        else:
            texts = args.query_texts()
            allowed = [t for t in texts if admit(t)]
            if not allowed:
                dropped.append(f"{call.name}: over the {max_queries}-query budget")
                continue
            if len(allowed) < len(texts) and call.name == "keyword_metrics":
                dropped.append(f"keyword_metrics: trimmed {len(texts)} keywords to "
                               f"{len(allowed)} to stay inside the query budget")
                call = ToolCall(name=call.name, id=call.id,
                                args={**call.args, "keywords": allowed})

        per_tool[call.name] = per_tool.get(call.name, 0) + 1
        kept.append(call)
    return kept, dropped


@node("plan_queries")
def plan_queries(state: PipelineState, deps: Deps) -> dict[str, Any]:
    profile = state["profile"]
    prompt = PLANNER_PROMPT.format(
        name=profile.name,
        domain=profile.domain,
        industry=profile.industry or "unspecified",
        competitors=", ".join(profile.competitors) or "unspecified",
        question=state["question"],
        max_queries=deps.settings.max_planned_queries,
        serp_limit=TOOL_CALL_LIMITS["google_serp"],
    )
    try:
        bound = deps.llm.bind_tools(TOOLS)
        message: AIMessage = bound.invoke([HumanMessage(content=prompt)])
    except Exception as exc:  # timeout / auth / rate limit -> fallback_plan
        log.warning("plan.llm_failed", extra={"error": str(exc)})
        return {
            "tool_calls": [],
            "errors": [PipelineError(node="plan_queries", kind="llm",
                                     message=f"{type(exc).__name__}: {exc}")],
            "_ok": False,
        }

    raw_calls = getattr(message, "tool_calls", None) or []
    calls, dropped = enforce_limits(
        [ToolCall(name=c.get("name", ""), args=c.get("args") or {}, id=c.get("id"))
         for c in raw_calls],
        deps.settings.max_planned_queries,
    )
    if dropped:
        log.warning("plan.over_budget", extra={"dropped": dropped})

    if not calls:
        reason = ("model returned no tool calls" if not raw_calls
                  else f"every planned call was rejected by the budget: {'; '.join(dropped)}")
        return {
            "tool_calls": [],
            "errors": [PipelineError(node="plan_queries", kind="llm", message=reason)],
            "_ok": False,
            "_tokens": tokens_from(message),
        }

    return {
        "tool_calls": calls,
        "_tokens": tokens_from(message),
        "_detail": {"planned_calls": [c.name for c in calls], "dropped_calls": dropped},
    }


@node("fallback_plan")
def fallback_plan(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Deterministic template built from the question, brand name and industry --
    the only fields a profile carries."""
    profile = state["profile"]
    calls, dropped = enforce_limits(
        [ToolCall(name=c["name"], args=c["args"], id=c["id"])
         for c in template_tool_calls(
             state["question"], name=profile.name, industry=profile.industry,
             model_name=deps.settings.llm_model,
             max_queries=deps.settings.max_planned_queries,
         )],
        deps.settings.max_planned_queries,
    )
    return {
        "tool_calls": calls,
        "degraded": True,
        "_detail": {"planned_calls": [c.name for c in calls], "dropped_calls": dropped},
    }


# --------------------------------------------------------------------------
# 2. retrieve -- validates and executes; never calls an LLM
# --------------------------------------------------------------------------

@node("retrieve")
def retrieve(state: PipelineState, deps: Deps) -> dict[str, Any]:
    payloads: list[RawPayload] = []
    errors: list[PipelineError] = []
    planned: dict[str, PlannedQuery] = {}
    retries = 0
    calls_before = deps.client.api_calls   # counts failed attempts too, not just successes
    serp_attempted = state.get("serp_attempted", False)

    for call in state.get("tool_calls", []):
        try:
            args = validate_args(call.name, call.args)
        except ToolArgumentError as exc:
            log.warning("retrieve.invalid_args", extra={"tool": call.name, "error": exc.message})
            errors.append(PipelineError(node="retrieve", kind="tool_argument",
                                        tool=call.name, message=exc.message))
            continue

        texts = args.query_texts()
        for text in texts:
            entry = planned.setdefault(
                query_key(text), PlannedQuery(query_key=query_key(text), query_text=text)
            )
            if call.name not in entry.tools:
                entry.tools.append(call.name)

        if call.name == "google_serp":
            serp_attempted = True

        try:
            execution = deps.client.execute(call.name, args)
        except ProviderError as exc:
            errors.append(PipelineError(
                node="retrieve", kind="provider", tool=call.name,
                query_keys=[query_key(t) for t in texts],
                classification=exc.classification, status_code=exc.status_code,
                message=exc.message, attempts=exc.attempts_made,
            ))
            retries += max(0, exc.attempts_made - 1)
            for text in texts:
                planned[query_key(text)].failed_tools.append(call.name)
            continue

        retries += execution.retries
        payloads.append(RawPayload(
            tool=call.name, query_texts=texts, payload=execution.payload,
            classification=execution.classification,
            usable=execution.classification in ("success", "partial_usable", "empty"),
        ))

    merged_planned = {p.query_key: p for p in state.get("planned_queries", [])}
    merged_planned.update(planned)

    return {
        "raw_payloads": payloads,
        "errors": errors,
        "planned_queries": list(merged_planned.values()),
        "serp_attempted": serp_attempted,
        "degraded": state.get("degraded", False) or bool(errors),
        "_ok": not errors,
        "_retries": retries,
        "_api_calls": deps.client.api_calls - calls_before,
        "_detail": {"payloads": len(payloads), "failed_calls": len(errors)},
    }


@node("fallback_serp")
def fallback_serp(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """One minimal SERP call for the case where the plan never included one.
    Retrying an already-failed dependency would be theatre, not a fallback."""
    planned = state.get("planned_queries") or []
    keyword = planned[0].query_text if planned else (core_phrase(state["question"]) or "brand")
    call = ToolCall(name="google_serp",
                    args={"keyword": keyword, "location_code": 2840, "language_code": "en",
                          "depth": 10, "load_async_ai_overview": False},
                    id="fallback_serp_0")
    update = retrieve.__wrapped__({**state, "tool_calls": [call]}, deps)
    update["degraded"] = True
    update["_detail"] = {**update.get("_detail", {}), "keyword": keyword}
    return update


# --------------------------------------------------------------------------
# 3. normalize -- raw JSON to typed rows, grouped by query_key
# --------------------------------------------------------------------------

def _same_domain(candidate: str | None, target: str) -> bool:
    if not candidate:
        return False
    a = candidate.lower().removeprefix("www.").rstrip("/")
    b = target.lower().removeprefix("https://").removeprefix("http://").removeprefix("www.")
    b = b.split("/")[0].rstrip("/")
    return bool(b) and (a == b or a.endswith("." + b))


def _extract_serp(payload: dict, texts: list[str], domain: str) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    for task in payload.get("tasks") or []:
        for result in task.get("result") or []:
            text = result.get("keyword") or (texts[0] if texts else "")
            if not text:
                continue
            items = result.get("items") or []
            organic = [i for i in items if i.get("type") == "organic"]
            position = None
            for item in organic:
                if _same_domain(item.get("domain"), domain):
                    position = item.get("rank_group") or item.get("rank_absolute")
                    break
            records.append(NormalizedRecord(
                query_key=query_key(text), query_text=text, source="organic",
                domain_visible=position is not None,
                visibility_position=position,
                evidence={"top_domains": [i.get("domain") for i in organic[:5]],
                          "results_inspected": len(organic)},
            ))
            for item in (i for i in items if i.get("type") == "ai_overview"):
                refs = [r.get("domain") for r in (item.get("references") or [])]
                records.append(NormalizedRecord(
                    query_key=query_key(text), query_text=text, source="ai_overview",
                    ai_overview_mentioned=any(_same_domain(d, domain) for d in refs),
                    evidence={"referenced_domains": refs[:8]},
                ))
    return records


def _extract_keyword_metrics(payload: dict, texts: list[str]) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    for task in payload.get("tasks") or []:
        for result in task.get("result") or []:
            for item in result.get("items") or []:
                text = item.get("keyword")
                if not text:
                    continue
                info = item.get("keyword_info") or {}
                props = item.get("keyword_properties") or {}
                records.append(NormalizedRecord(
                    query_key=query_key(text), query_text=text, source="keyword_metrics",
                    search_volume=info.get("search_volume"),
                    competitive_difficulty=props.get("keyword_difficulty"),
                    evidence={"cpc": info.get("cpc"), "competition": info.get("competition")},
                ))
    return records


def _extract_chatgpt(payload: dict, texts: list[str], domain: str,
                     brand: str) -> list[NormalizedRecord]:
    text = texts[0] if texts else ""
    if not text:
        return []
    chunks: list[str] = []
    for task in payload.get("tasks") or []:
        for result in task.get("result") or []:
            for item in result.get("items") or []:
                for section in item.get("sections") or []:
                    if isinstance(section.get("text"), str):
                        chunks.append(section["text"])
    answer = "\n".join(chunks)
    haystack = answer.lower()
    bare = domain.lower().removeprefix("www.").split(".")[0]
    mentioned = _same_domain_in_text(haystack, domain) or (
        bool(brand) and brand.lower() in haystack) or (bool(bare) and bare in haystack)
    return [NormalizedRecord(
        query_key=query_key(text), query_text=text, source="chatgpt",
        chatgpt_mentioned=mentioned,
        evidence={"answer_excerpt": answer[:400]},
    )]


def _same_domain_in_text(haystack: str, domain: str) -> bool:
    bare = domain.lower().removeprefix("www.")
    return bool(bare) and bare in haystack


@node("normalize")
def normalize(state: PipelineState, deps: Deps) -> dict[str, Any]:
    profile = state["profile"]
    records: list[NormalizedRecord] = []
    for raw in state.get("raw_payloads", []):
        if raw.tool == "google_serp":
            records += _extract_serp(raw.payload, raw.query_texts, profile.domain)
        elif raw.tool == "keyword_metrics":
            records += _extract_keyword_metrics(raw.payload, raw.query_texts)
        elif raw.tool == "chatgpt_response":
            records += _extract_chatgpt(raw.payload, raw.query_texts,
                                        profile.domain, profile.name)

    by_key: dict[str, list[NormalizedRecord]] = {}
    for record in records:
        by_key.setdefault(record.query_key, []).append(record)

    # A row for every distinct planned query, including ones whose retrieval failed --
    # a failed query is exactly the one most worth rechecking (PLAN §6.3).
    merged: list[MergedQuery] = []
    for plan in state.get("planned_queries", []):
        rows = by_key.pop(plan.query_key, [])
        merged.append(_merge(plan.query_key, plan.query_text, rows, plan))
    for key, rows in by_key.items():
        merged.append(_merge(key, rows[0].query_text, rows, None))

    return {
        "normalized": records,
        "merged": merged,
        "_detail": {"records": len(records), "queries": len(merged)},
    }


def _merge(key: str, text: str, rows: list[NormalizedRecord],
           plan: PlannedQuery | None) -> MergedQuery:
    merged = MergedQuery(query_key=key, query_text=text)
    merged.sources = sorted({r.source for r in rows})
    for field in ("search_volume", "competitive_difficulty", "domain_visible",
                  "visibility_position", "ai_overview_mentioned", "chatgpt_mentioned"):
        for row in rows:
            value = getattr(row, field)
            if value is not None:
                setattr(merged, field, value)
                break
    merged.evidence = {r.source: r.evidence for r in rows if r.evidence}
    if plan is not None and plan.failed_tools:
        # A call that returned zero rows successfully (status 40102) is not a failure.
        merged.retrieval_status = "partial" if rows else "failed"
    else:
        merged.retrieval_status = "ok"
    return merged


# --------------------------------------------------------------------------
# 4. analyze -- the only synthesis node
# --------------------------------------------------------------------------

ANALYST_PROMPT = """You are a search-visibility analyst.

BRAND: {name}
DOMAIN: {domain}
QUESTION: {question}

QUERIES_JSON: {rows}

Each row already carries a deterministic opportunity_score between 0 and 1. Do not
recompute or contradict it. Explain what the data means for this brand and propose
content actions for the biggest gaps.

Respond with ONLY a JSON object of this exact shape:
{{"summary": "...",
  "insights": [{{"query_key": "...", "rationale": "..."}}],
  "recommendations": [{{"target_query_key": "...", "content_type": "...", "title": "...",
                        "rationale": "...", "target_keywords": ["..."],
                        "priority": "high|medium|low"}}]}}"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_analysis(content: str) -> LLMAnalysis:
    text = _FENCE.sub("", content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model response contained no JSON object")
    return LLMAnalysis.model_validate(json.loads(text[start:end + 1]))


def _rows_for_prompt(merged: list[MergedQuery]) -> str:
    return json.dumps([
        {"query_key": m.query_key, "search_volume": m.search_volume,
         "competitive_difficulty": m.competitive_difficulty,
         "domain_visible": m.domain_visible, "visibility_position": m.visibility_position,
         "ai_overview_mentioned": m.ai_overview_mentioned,
         "chatgpt_mentioned": m.chatgpt_mentioned,
         "retrieval_status": m.retrieval_status,
         "opportunity_score": m.opportunity_score}
        for m in merged
    ])


def _insight(row: MergedQuery, rationale: str) -> Insight:
    return Insight(
        query_key=row.query_key, query_text=row.query_text,
        opportunity_score=row.opportunity_score, domain_visible=row.domain_visible,
        visibility_position=row.visibility_position, search_volume=row.search_volume,
        competitive_difficulty=row.competitive_difficulty, rationale=rationale,
    )


def _templated_rationale(row: MergedQuery, brand: str) -> str:
    if row.retrieval_status == "failed":
        return f"Retrieval failed for '{row.query_text}'; no visibility evidence collected."
    if row.domain_visible is None:
        where = "organic standing unknown"
    elif row.domain_visible:
        where = f"ranking at position {row.visibility_position or 'unknown'}"
    else:
        where = "absent from the inspected organic results"
    return (f"{brand} is {where} for '{row.query_text}' "
            f"(volume {row.search_volume}, difficulty {row.competitive_difficulty}).")


@node("analyze")
def analyze(state: PipelineState, deps: Deps) -> dict[str, Any]:
    profile = state["profile"]
    merged = apply_scores(state.get("merged", []))
    prompt = ANALYST_PROMPT.format(
        name=profile.name, domain=profile.domain, question=state["question"],
        rows=_rows_for_prompt(merged),
    )
    try:
        message = deps.llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_analysis(
            message.content if isinstance(message.content, str) else str(message.content)
        )
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        log.warning("analyze.invalid_output", extra={"error": str(exc)})
        return {"analysis": None, "merged": merged, "_ok": False,
                "errors": [PipelineError(node="analyze", kind="schema",
                                         message=f"analysis failed validation: {exc}")]}
    except Exception as exc:
        log.warning("analyze.llm_failed", extra={"error": str(exc)})
        return {"analysis": None, "merged": merged, "_ok": False,
                "errors": [PipelineError(node="analyze", kind="llm",
                                         message=f"{type(exc).__name__}: {exc}")]}

    rationales = {i.query_key: i.rationale for i in parsed.insights}
    by_key = {m.query_key: m for m in merged}
    insights = [
        _insight(row, rationales.get(row.query_key)
                 or _templated_rationale(row, profile.name))
        for row in sorted(merged, key=lambda m: m.opportunity_score, reverse=True)
    ]
    recommendations = [
        Recommendation(
            target_query_key=r.target_query_key, content_type=r.content_type,
            title=r.title, rationale=r.rationale,
            target_keywords=r.target_keywords or [r.target_query_key],
            priority=r.priority,
        )
        for r in parsed.recommendations if r.target_query_key in by_key
    ]
    analysis = AnalysisResult(summary=parsed.summary, insights=insights,
                              recommendations=recommendations, generated_by="llm")
    return {
        "analysis": analysis, "merged": merged,
        "_tokens": tokens_from(message),
        "_detail": {"insights": len(insights), "recommendations": len(recommendations)},
    }


@node("fallback_analysis")
def fallback_analysis(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Score-only analysis so `report` always receives a valid AnalysisResult."""
    profile = state["profile"]
    merged = apply_scores(state.get("merged", []))
    ranked = sorted(merged, key=lambda m: m.opportunity_score, reverse=True)
    insights = [_insight(row, _templated_rationale(row, profile.name)) for row in ranked]
    recommendations = [
        Recommendation(
            target_query_key=row.query_key,
            content_type=content_type_for(row.domain_visible),
            title=f"{row.query_text.title()}: buyer's guide",
            rationale=(f"Opportunity score {row.opportunity_score} with "
                       f"{'no' if row.domain_visible is False else 'partial'} organic "
                       f"visibility."),
            target_keywords=[row.query_text],
            priority="high" if row.opportunity_score >= 0.6 else "medium",
        )
        for row in ranked[:3] if row.retrieval_status != "failed"
    ]
    summary = (f"Deterministic analysis of {len(merged)} queries for {profile.name}. "
               f"Narrative synthesis was unavailable, so scores and rankings are reported "
               f"without model commentary.")
    return {
        "analysis": AnalysisResult(summary=summary, insights=insights,
                                   recommendations=recommendations,
                                   generated_by="deterministic"),
        "merged": merged,
        "degraded": True,
        "_detail": {"insights": len(insights)},
    }


# --------------------------------------------------------------------------
# 5. report -- deterministic assembly, no LLM
# --------------------------------------------------------------------------

def run_status(state: PipelineState) -> str:
    if not state.get("merged"):
        return "partial"
    if state.get("errors") or state.get("degraded"):
        return "partial"
    return "completed"


@node("report")
def report(state: PipelineState, deps: Deps) -> dict[str, Any]:
    profile = state["profile"]
    analysis = state.get("analysis")
    merged = state.get("merged", [])
    errors = state.get("errors", [])
    status = run_status(state)

    visible = [m for m in merged if m.domain_visible is True]
    invisible = [m for m in merged if m.domain_visible is False]
    unknown = [m for m in merged if m.domain_visible is None]
    ranked = sorted(merged, key=lambda m: m.opportunity_score, reverse=True)

    lines = [
        f"{profile.name} ({profile.domain}) — {len(merged)} queries analysed for: "
        f"\"{state['question']}\".",
        f"Organic visibility: {len(visible)} visible, {len(invisible)} not visible, "
        f"{len(unknown)} unknown.",
    ]
    # A query whose retrieval failed still scores (0.295 on pure defaults), so with
    # everything down every row ties and "largest" would be arbitrary sort order
    # presented as a finding.
    measured = [m for m in ranked if m.retrieval_status != "failed"]
    if measured:
        top = measured[0]
        lines.append(f"Largest opportunity: '{top.query_text}' at score "
                     f"{top.opportunity_score}.")
    elif ranked:
        lines.append("No query returned usable data, so the scores below are defaults "
                     "and no opportunity ranking is meaningful.")
    if analysis and analysis.summary:
        lines.append(analysis.summary)
    if errors:
        lines.append(f"Ran degraded: {len(errors)} retrieval or model failure(s) recorded; "
                     f"results below cover what was retrieved.")

    document = {
        "question": state["question"],
        "profile": {"uuid": profile.uuid, "name": profile.name, "domain": profile.domain,
                    "industry": profile.industry},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "correlation_id": state.get("correlation_id"),
        "status": status,
        "degraded": bool(state.get("degraded")),
        "llm_mode": llm_mode(deps.llm),
        "analysis_generated_by": analysis.generated_by if analysis else "none",
        "queries_analysed": len(merged),
        "visibility": {"visible": len(visible), "not_visible": len(invisible),
                       "unknown": len(unknown)},
        "queries": [m.model_dump() for m in ranked],
        "insights": [i.model_dump() for i in (analysis.insights if analysis else [])],
        "recommendations": [r.model_dump()
                            for r in (analysis.recommendations if analysis else [])],
        "errors": [e.model_dump() for e in errors],
        "summary": " ".join(lines),
    }
    return {"report_document": document, "_detail": {"status": status}}
