"""DataForSEO client, the three tool definitions bound to the planner LLM, and the
argument-validation gate that sits between the model and any paid HTTP call.

Endpoints and every field limit were verified against the live docs (PLAN §14).
"""
from __future__ import annotations

import base64
from functools import lru_cache
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.observability.logging import get_logger
from app.resilience import (
    ProviderError,
    ToolArgumentError,
    classify_http_status,
    classify_response,
    retry_with_backoff,
)
from app.schemas import TOOL_ARGS, ChatGptResponseArgs, GoogleSerpArgs, KeywordMetricsArgs
from app.tools.mock import MockBackend

log = get_logger(__name__)

ENDPOINTS = {
    "google_serp": "/v3/serp/google/organic/live/advanced",
    "keyword_metrics": "/v3/dataforseo_labs/google/keyword_overview/live",
    "chatgpt_response": "/v3/ai_optimization/chat_gpt/llm_responses/live",
}


class ToolExecution(BaseModel):
    tool: str
    payload: dict[str, Any]
    classification: str
    retries: int = 0
    api_calls: int = 0


def validate_args(tool: str, raw: dict[str, Any]) -> BaseModel:
    """The gate. A hallucinated field, a missing one, or a limit breach fails here --
    locally, before a paid call, with an error the graph can route on."""
    model = TOOL_ARGS.get(tool)
    if model is None:
        raise ToolArgumentError(tool, f"unknown tool {tool!r}")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ToolArgumentError(tool, problems) from exc


def request_body(tool: str, args: BaseModel) -> list[dict[str, Any]]:
    """DataForSEO takes an array of task objects. query_text is ours, not theirs."""
    data = args.model_dump(exclude_none=True, exclude={"query_text"})
    return [data]


class HttpBackend:
    def __init__(self, settings: Settings) -> None:
        token = base64.b64encode(
            f"{settings.dataforseo_login}:{settings.dataforseo_password}".encode()
        ).decode()
        self._headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
        self._base_url = settings.dataforseo_base_url

    def post(self, path: str, body: list[dict[str, Any]], timeout: float, tool: str):
        try:
            response = httpx.post(
                f"{self._base_url}{path}", json=body, headers=self._headers, timeout=timeout
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as exc:
            raise ProviderError(
                f"transport failure: {type(exc).__name__}: {exc}",
                classification="retryable", tool=tool,
            ) from exc
        try:
            return response.status_code, response.json()
        except ValueError as exc:
            raise ProviderError(
                f"non-JSON response (HTTP {response.status_code})",
                classification="retryable" if response.status_code >= 500 else "terminal",
                status_code=response.status_code, tool=tool,
            ) from exc


class DataForSEOClient:
    """Executes one validated tool call, with classification and retry around it."""

    def __init__(self, settings: Settings | None = None, backend: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = backend or self._default_backend()
        self.api_calls = 0

    def _default_backend(self) -> Any:
        base = (MockBackend(latency_ms=self.settings.mock_latency_ms)
                if self.settings.mock_dataforseo else HttpBackend(self.settings))
        if self.settings.serpapi_api_key:
            # SerpApi answers google_serp; keyword_metrics and chatgpt_response have no
            # SerpApi equivalent and fall through to `base`.
            from app.tools.serpapi import SerpApiBackend

            return SerpApiBackend(self.settings, fallback=base)
        return base

    def timeout_for(self, tool: str) -> float:
        # Per-tool, not one global: the ChatGPT live endpoint is documented at up to 120s.
        if tool == "chatgpt_response":
            return self.settings.chatgpt_timeout_seconds
        return self.settings.http_timeout_seconds

    def execute(self, tool: str, args: BaseModel) -> ToolExecution:
        path = ENDPOINTS[tool]
        body = request_body(tool, args)
        timeout = self.timeout_for(tool)
        calls_before = self.api_calls

        def attempt() -> tuple[dict[str, Any], str]:
            self.api_calls += 1
            status, payload = self.backend.post(path, body, timeout, tool)

            http_class = classify_http_status(status)
            if http_class != "success":
                # The body of a non-2xx usually carries the actionable reason (e.g. 40104
                # "Please verify your account"). Reporting only "HTTP 403" throws that away
                # and leaves whoever is debugging with nothing to act on.
                detail = ""
                body_code, body_message = payload.get("status_code"), payload.get("status_message")
                if body_code or body_message:
                    detail = f" - {body_code}: {body_message}"
                raise ProviderError(
                    f"HTTP {status}{detail}", classification=http_class,
                    status_code=status, tool=tool,
                )

            # Layer two: DataForSEO returns most failures inside an HTTP 200.
            classification, code, message = classify_response(payload)
            if classification in ("retryable", "terminal"):
                raise ProviderError(
                    message, classification=classification, status_code=code, tool=tool
                )
            log.info(
                "tool.call.ok",
                extra={"tool": tool, "classification": classification,
                       "status_code": code, "timeout_seconds": timeout},
            )
            return payload, classification

        (payload, classification), retries = retry_with_backoff(
            attempt,
            attempts=self.settings.retry_max_attempts,
            base_delay=self.settings.retry_base_delay_seconds,
            max_delay=self.settings.retry_max_delay_seconds,
            jitter=self.settings.retry_jitter,
            context={"tool": tool},
        )
        return ToolExecution(
            tool=tool, payload=payload, classification=classification,
            retries=retries, api_calls=self.api_calls - calls_before,
        )


@lru_cache
def get_client() -> DataForSEOClient:
    return DataForSEOClient()


# --------------------------------------------------------------------------
# Tool definitions -- these are what bind_tools() advertises to the planner.
# --------------------------------------------------------------------------

def _run(tool: str, **kwargs: Any) -> dict[str, Any]:
    return get_client().execute(tool, validate_args(tool, kwargs)).payload


def _make(tool: str, description: str, args_schema: type[BaseModel]) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda **kw: _run(tool, **kw),
        name=tool,
        description=description,
        args_schema=args_schema,
    )


google_serp = _make(
    "google_serp",
    "Fetch live Google organic search results for one keyword, including the AI Overview "
    "block when load_async_ai_overview is true. Use this to find out who currently ranks "
    "for a query and whether a domain appears.",
    GoogleSerpArgs,
)

keyword_metrics = _make(
    "keyword_metrics",
    "Fetch monthly search volume and keyword difficulty (0-100) for up to 700 keywords. "
    "Use this to size the demand behind a query and how hard it is to rank for.",
    KeywordMetricsArgs,
)

chatgpt_response = _make(
    "chatgpt_response",
    "Ask ChatGPT a question and capture its answer, to check which brands an LLM recommends "
    "for a topic. user_prompt is limited to 500 characters.",
    ChatGptResponseArgs,
)

TOOLS = [google_serp, keyword_metrics, chatgpt_response]
