"""SerpApi transport for `google_serp`, adapted into the DataForSEO response envelope.

Everything downstream -- `normalize`, scoring, persistence, the report -- reads the
DataForSEO shape. Translating at the transport boundary means one adapter here instead of
a second code path through the whole pipeline, and every test of that pipeline keeps
covering both providers.

SerpApi sells search results only. It has no keyword-volume and no ChatGPT endpoint, so
those two tools fall through to whatever DataForSEO transport is configured (mock by
default). That split is reported by `/health` and recorded in the run config, because a
run whose rankings are real and whose volumes are fixtures must not look like one where
both are real.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.observability.logging import get_logger
from app.resilience import ProviderError

log = get_logger(__name__)

ENDPOINT = "https://serpapi.com/search.json"

# SerpApi reports failure as a JSON `error` string, with no code. The pipeline classifies
# on DataForSEO status codes, so each recognised phrase maps onto the code that carries
# the same meaning; anything unrecognised is terminal rather than optimistically retried.
_EMPTY_MARKERS = ("hasn't returned any results", "fully empty")
_EMPTY_CODE = 40102          # "No Search Results" -- usable, just nothing to report
_TERMINAL_CODE = 40100       # auth / quota / malformed request
_OK_CODE = 20000


def domain_of(url: str | None) -> str | None:
    """SerpApi returns a link; the pipeline matches on a bare domain."""
    if not url:
        return None
    host = urlparse(url if "//" in url else f"https://{url}").netloc.lower()
    return host.removeprefix("www.").split(":")[0] or None


def to_envelope(payload: dict[str, Any], keyword: str) -> dict[str, Any]:
    """One SerpApi response -> the DataForSEO body `_extract_serp` already understands."""
    error = payload.get("error")
    if error:
        empty = any(marker in error.lower() for marker in _EMPTY_MARKERS)
        code = _EMPTY_CODE if empty else _TERMINAL_CODE
        return {"status_code": code, "status_message": error,
                "tasks": [{"status_code": code, "status_message": error, "result": None}]}

    items: list[dict[str, Any]] = []
    for entry in payload.get("organic_results") or []:
        items.append({
            "type": "organic",
            "rank_group": entry.get("position"),
            "rank_absolute": entry.get("position"),
            "domain": domain_of(entry.get("link")),
            "title": entry.get("title"),
            "url": entry.get("link"),
        })

    # ponytail: inline AI Overview references only. SerpApi often returns just a
    # `page_token` and needs a second billed search (engine=google_ai_overview) to
    # expand it; spending a second search per query is not worth it on a 250-search
    # plan. If AI Overview coverage matters, follow the token here.
    overview = payload.get("ai_overview") or {}
    references = overview.get("references") or []
    if references:
        items.append({
            "type": "ai_overview",
            "references": [{"domain": domain_of(r.get("link")) or r.get("source"),
                            "url": r.get("link")} for r in references],
        })
    elif overview.get("page_token"):
        log.info("serpapi.ai_overview_deferred", extra={"keyword": keyword})

    return {
        "status_code": _OK_CODE, "status_message": "Ok.",
        "tasks": [{
            "status_code": _OK_CODE, "status_message": "Ok.",
            "result": [{"keyword": keyword, "items": items,
                        "se_results_count": payload.get("search_information", {})
                        .get("total_results")}],
        }],
    }


class SerpApiBackend:
    """Speaks the same `post(path, body, timeout, tool)` contract as the other backends."""

    def __init__(self, settings: Settings, fallback: Any) -> None:
        self._key = settings.serpapi_api_key
        self._location_code = None
        self.fallback = fallback

    def post(self, path: str, body: list[dict[str, Any]], timeout: float, tool: str):
        if tool != "google_serp":
            return self.fallback.post(path, body, timeout, tool)

        task = body[0] if body else {}
        params = {
            "engine": "google",
            "q": task.get("keyword", ""),
            "num": task.get("depth", 10),
            "hl": task.get("language_code", "en"),
            "gl": "us",                       # 2840 is the only location this project uses
            "device": task.get("device", "desktop"),
            "api_key": self._key,
        }
        try:
            response = httpx.get(ENDPOINT, params=params, timeout=timeout)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as exc:
            raise ProviderError(
                f"transport failure: {type(exc).__name__}: {exc}",
                classification="retryable", tool=tool,
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"non-JSON response (HTTP {response.status_code})",
                classification="retryable" if response.status_code >= 500 else "terminal",
                status_code=response.status_code, tool=tool,
            ) from exc

        envelope = to_envelope(payload, task.get("keyword", ""))
        if response.status_code != 200:
            # Carry SerpApi's own words up: "HTTP 401" alone tells nobody whether the key
            # is wrong or the 250 searches are spent.
            envelope.setdefault("status_message", payload.get("error", ""))
        return response.status_code, envelope
