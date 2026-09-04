"""LLM access, plus the offline seam.

langchain-core's fake chat models do not implement bind_tools(), so the assessed
tool-calling path could not be exercised without a key. ScriptedToolCallingLLM does
implement it and returns real AIMessage(tool_calls=[...]) objects, so offline runs and
tests travel exactly the same code path as OpenAI -- including deliberately malformed
tool calls and raised provider errors.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from app.config import Settings, get_settings

# --------------------------------------------------------------------------
# Deterministic planner template.
# Shared by the offline model and by fallback_plan, so the no-key path and the
# LLM-failed path produce the same well-formed calls instead of two near-copies.
# --------------------------------------------------------------------------

_STOP = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "in", "is", "me", "my",
    "of", "on", "our", "should", "the", "to", "we", "what", "which", "who", "why",
    "where", "when", "did", "will", "would", "could", "much", "many", "am",
    "i", "it", "that", "this", "with", "about", "there", "any",
    # phrasing of the research question itself, not part of the search query
    "visible", "visibility", "rank", "ranking", "ranks", "appear", "appearing",
    "us", "you", "your", "have", "has", "been", "get", "find", "show", "showing",
    "mentioned", "mention", "currently", "search", "searches", "results",
}


def _clip(phrase: str) -> str:
    """Respect the documented keyword limits: <=80 chars and <=10 words."""
    words = phrase.split()[:10]
    out = " ".join(words)
    while len(out) > 80 and words:
        words.pop()
        out = " ".join(words)
    return out


def core_phrase(question: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9+&']+", question.lower()) if w not in _STOP]
    return _clip(" ".join(words[:6]))


def query_variants(question: str, *, name: str = "", industry: str | None = None,
                   limit: int = 4) -> list[str]:
    core = core_phrase(question) or (name.lower() if name else "brand visibility")
    core_words = set(core.split())
    # A head term next to the long-tail one: visibility usually differs sharply between them.
    head = _clip(" ".join(core.split()[:3]))
    candidates = [core]
    if head != core:
        candidates.append(head)
    if industry and not set(industry.lower().split()) & core_words:
        candidates.append(_clip(f"{core} {industry.lower()}"))
    if not core.startswith("best"):
        candidates.append(_clip(f"best {core}"))
    candidates.append(_clip(f"{core} alternatives"))
    seen, out = set(), []
    for candidate in candidates:
        clipped = _clip(candidate)
        if clipped and clipped not in seen:
            seen.add(clipped)
            out.append(clipped)
    return out[:limit]


def template_tool_calls(
    question: str, *, name: str = "", industry: str | None = None,
    model_name: str = "gpt-4o-mini", max_queries: int = 8,
) -> list[dict[str, Any]]:
    """A complete, schema-valid plan derived from the question, profile name and industry.

    Profiles carry no keyword field, so those three are the only inputs available.
    """
    variants = query_variants(question, name=name, industry=industry,
                              limit=min(4, max(1, max_queries)))
    calls: list[dict[str, Any]] = []
    for i, variant in enumerate(variants[:2]):   # SERP is the expensive one; cap at 2
        calls.append({
            "name": "google_serp",
            "args": {"keyword": variant, "location_code": 2840, "language_code": "en",
                     "depth": 10, "load_async_ai_overview": True},
            "id": f"call_serp_{i}",
            "type": "tool_call",
        })
    calls.append({
        "name": "keyword_metrics",
        "args": {"keywords": variants, "location_code": 2840, "language_code": "en"},
        "id": "call_kw_0",
        "type": "tool_call",
    })
    calls.append({
        "name": "chatgpt_response",
        "args": {
            "query_text": variants[0],
            "user_prompt": _clip_chars(f"Which brands or products would you recommend for "
                                       f"'{variants[0]}'? Name the top options.", 500),
            "model_name": model_name,
            "max_output_tokens": 1024,
        },
        "id": "call_llm_0",
        "type": "tool_call",
    })
    return calls


def content_type_for(domain_visible: bool | None) -> str:
    if domain_visible is False:
        return "comparison guide"
    if domain_visible is True:
        return "refresh existing page"
    return "visibility audit"


def _clip_chars(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "."


# --------------------------------------------------------------------------
# Offline model
# --------------------------------------------------------------------------

_QUESTION_RE = re.compile(r"^QUESTION:\s*(.+)$", re.MULTILINE)
_NAME_RE = re.compile(r"^BRAND:\s*(.+)$", re.MULTILINE)
_INDUSTRY_RE = re.compile(r"^INDUSTRY:\s*(.+)$", re.MULTILINE)
_QUERIES_RE = re.compile(r"^QUERIES_JSON:\s*(\[.*\])\s*$", re.MULTILINE)


class ScriptedToolCallingLLM(BaseChatModel):
    """Deterministic stand-in that implements bind_tools().

    With no script it reads the structured prompt and answers sensibly, so the whole
    system runs with no API key. Tests hand it explicit scripts to force malformed tool
    calls, schema-violating analysis, or a raised provider error.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_script: list[list[dict[str, Any]]] = Field(default_factory=list)
    content_script: list[str] = Field(default_factory=list)
    raise_on_invoke: Exception | None = None
    bound_tools: list[Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedToolCallingLLM":
        # Shares the script lists by reference, so a copy keeps consuming the same script.
        return self.model_copy(update={"bound_tools": list(tools)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.raise_on_invoke is not None:
            raise self.raise_on_invoke

        prompt = "\n".join(
            m.content for m in messages if isinstance(getattr(m, "content", None), str)
        )
        if self.bound_tools:
            calls = (self.tool_call_script.pop(0) if self.tool_call_script
                     else self._derive_tool_calls(prompt))
            message = AIMessage(content="", tool_calls=calls)
        else:
            content = (self.content_script.pop(0) if self.content_script
                       else self._derive_analysis(prompt))
            message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- offline behaviour -------------------------------------------------

    @staticmethod
    def _derive_tool_calls(prompt: str) -> list[dict[str, Any]]:
        question = (_QUESTION_RE.search(prompt) or [None, ""])[1].strip()
        name = (_NAME_RE.search(prompt) or [None, ""])[1].strip()
        industry = (_INDUSTRY_RE.search(prompt) or [None, ""])[1].strip() or None
        return template_tool_calls(question, name=name, industry=industry)

    @staticmethod
    def _derive_analysis(prompt: str) -> str:
        match = _QUERIES_RE.search(prompt)
        rows = json.loads(match.group(1)) if match else []
        brand = (_NAME_RE.search(prompt) or [None, "the brand"])[1].strip() or "the brand"
        insights, recommendations = [], []
        for row in rows:
            key = row.get("query_key", "")
            visible = row.get("domain_visible")
            where = ("ranking at position %s" % row.get("visibility_position")
                     if visible else "absent from the first page" if visible is False
                     else "of unknown organic standing")
            insights.append({
                "query_key": key,
                "rationale": (f"{brand} is {where} for '{key}', against monthly volume "
                              f"{row.get('search_volume')} and difficulty "
                              f"{row.get('competitive_difficulty')}."),
            })
        for row in sorted(rows, key=lambda r: r.get("opportunity_score", 0), reverse=True)[:3]:
            key = row.get("query_key", "")
            recommendations.append({
                "target_query_key": key,
                "content_type": content_type_for(row.get("domain_visible")),
                "title": f"{key.title()}: buyer's guide",
                "rationale": (f"Opportunity score {row.get('opportunity_score')} — the "
                              f"largest gap between demand and current visibility."),
                "target_keywords": [key],
                "priority": "high" if row.get("opportunity_score", 0) >= 0.6 else "medium",
            })
        gaps = sum(1 for r in rows if r.get("domain_visible") is False)
        return json.dumps({
            "summary": (f"Reviewed {len(rows)} queries for {brand}. "
                        f"{gaps} {'query has' if gaps == 1 else 'queries have'} no organic "
                        f"visibility and {'is' if gaps == 1 else 'are'} the clearest "
                        f"content gap{'' if gaps == 1 else 's'}."),
            "insights": insights,
            "recommendations": recommendations,
        })


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    if s.openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.llm_model,
            temperature=s.llm_temperature,
            timeout=s.llm_timeout_seconds,
            api_key=s.openai_api_key,
        )
    return ScriptedToolCallingLLM()


def llm_mode(settings: Settings | None = None) -> str:
    return "openai" if (settings or get_settings()).openai_api_key else "scripted"


def tokens_from(message: BaseMessage) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0) or 0)


def single_query_tool_calls(query_text: str, *, model_name: str = "gpt-4o-mini",
                            ) -> list[dict[str, Any]]:
    """Deterministic reconstruction of one query's calls, used by recheck (no planner LLM)."""
    keyword = _clip(query_text)
    return [
        {"name": "google_serp",
         "args": {"keyword": keyword, "location_code": 2840, "language_code": "en",
                  "depth": 10, "load_async_ai_overview": True},
         "id": "recheck_serp_0", "type": "tool_call"},
        {"name": "keyword_metrics",
         "args": {"keywords": [keyword], "location_code": 2840, "language_code": "en"},
         "id": "recheck_kw_0", "type": "tool_call"},
        {"name": "chatgpt_response",
         "args": {"query_text": keyword,
                  "user_prompt": _clip_chars(
                      f"Which brands or products would you recommend for '{keyword}'? "
                      f"Name the top options.", 500),
                  "model_name": model_name, "max_output_tokens": 1024},
         "id": "recheck_llm_0", "type": "tool_call"},
    ]
