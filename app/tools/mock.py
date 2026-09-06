"""Offline DataForSEO transport.

Serves the real response envelopes from fixtures/ and fills in the variable fields
deterministically from a CRC of the keyword, so every run of the demo and every test
produces the same numbers. Failure injection is `fail_first_n` per tool -- exact, not
random: a flaky test suite is worse than no test.

`domain_hint` is a mock-only affordance. A real SERP request carries no domain, so the
mock is told which domain to sometimes rank in order to produce both visible and
not-visible queries in the demo.
"""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"

# Padding for the fixture SERP: domains that plausibly rank for almost any query. The
# profile's own competitors go in front of these, so a coffee brand is not measured
# against Asana just because the fixture was first written for a project-management demo.
_GENERIC_POOL = [
    "reddit.com", "wikipedia.org", "quora.com", "youtube.com",
    "trustpilot.com", "medium.com", "forbes.com", "nytimes.com",
]


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _seed(text: str) -> int:
    return zlib.crc32(text.lower().strip().encode())


def _derive(keyword: str) -> dict[str, Any]:
    """Stable pseudo-metrics for one keyword."""
    s = _seed(keyword)
    return {
        "search_volume": [110, 480, 1300, 2900, 6600, 14800, 33100][s % 7],
        "difficulty": (s >> 3) % 101,
        "visible": (s >> 5) % 10 < 6,          # ~60% of keywords rank the hint domain
        "position": ((s >> 7) % 9) + 1,
        "ai_mentions": (s >> 11) % 10 < 4,
        "chatgpt_mentions": (s >> 13) % 10 < 5,
    }


class MockBackend:
    def __init__(
        self,
        *,
        domain_hint: str | None = None,
        competitors: list[str] | None = None,
        fail_first_n: dict[str, int] | None = None,
        latency_ms: int = 0,
        empty_for: set[str] | None = None,
    ) -> None:
        self.domain_hint = (domain_hint or "").lower().removeprefix("www.") or None
        self.pool = [c.lower().removeprefix("www.") for c in (competitors or [])]
        self.pool += [d for d in _GENERIC_POOL if d not in self.pool]
        self.fail_first_n = dict(fail_first_n or {})
        self.latency_ms = latency_ms
        self.empty_for = {k.lower() for k in (empty_for or set())}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, body: list[dict[str, Any]], timeout: float, tool: str):
        self.calls.append((tool, body[0] if body else {}))
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)

        remaining = self.fail_first_n.get(tool, 0)
        if remaining > 0:
            self.fail_first_n[tool] = remaining - 1
            return 200, _load("error_retryable")

        task = body[0] if body else {}
        if tool == "google_serp":
            return 200, self._serp(task)
        if tool == "keyword_metrics":
            return 200, self._keyword_overview(task)
        if tool == "chatgpt_response":
            return 200, self._chatgpt(task)
        return 404, {"status_code": 40400, "status_message": "Not Found.", "tasks": []}

    # -- builders ---------------------------------------------------------

    def _serp(self, task: dict[str, Any]) -> dict[str, Any]:
        keyword = task.get("keyword", "")
        if keyword.lower() in self.empty_for:
            return _load("error_empty")

        d = _derive(keyword)
        depth = int(task.get("depth", 10))
        env = _load("serp_envelope")
        result = env["tasks"][0]["result"][0]
        result["keyword"] = keyword
        result["location_code"] = task.get("location_code", 2840)
        result["language_code"] = task.get("language_code", "en")
        result["check_url"] = f"https://www.google.com/search?q={keyword.replace(' ', '+')}"
        result["se_results_count"] = d["search_volume"] * 9137

        domains = list(self.pool)
        if self.domain_hint and d["visible"]:
            slot = min(d["position"], depth) - 1
            domains.insert(slot, self.domain_hint)

        items: list[dict[str, Any]] = []
        for rank, domain in enumerate(domains[:depth], start=1):
            items.append({
                "type": "organic",
                "rank_group": rank,
                "rank_absolute": rank,
                "domain": domain,
                "title": f"{keyword.title()} - {domain}",
                "url": f"https://{domain}/{keyword.replace(' ', '-')}",
                "description": f"A guide to {keyword} from {domain}.",
            })

        if task.get("load_async_ai_overview"):
            referenced = domains[:3]
            if self.domain_hint and d["ai_mentions"] and self.domain_hint not in referenced:
                referenced.append(self.domain_hint)
            refs = [
                {"type": "ai_overview_reference", "domain": dom,
                 "url": f"https://{dom}/", "title": dom}
                for dom in referenced
            ]
            items.append({
                "type": "ai_overview",
                "rank_group": 1,
                "rank_absolute": 0,
                "asynchronous_ai_overview": True,
                "items": [{
                    "type": "ai_overview_element",
                    "text": f"Popular options for {keyword} include "
                            + ", ".join(referenced) + ".",
                    "references": refs,
                }],
                "references": refs,
            })

        result["items"] = items
        result["items_count"] = len(items)
        result["item_types"] = sorted({i["type"] for i in items})
        return env

    def _keyword_overview(self, task: dict[str, Any]) -> dict[str, Any]:
        env = _load("keyword_overview_envelope")
        result = env["tasks"][0]["result"][0]
        result["location_code"] = task.get("location_code", 2840)
        result["language_code"] = task.get("language_code", "en")
        items = []
        for kw in task.get("keywords", []):
            d = _derive(kw)
            items.append({
                "se_type": "google",
                "keyword": kw.lower(),
                "location_code": result["location_code"],
                "language_code": result["language_code"],
                "keyword_info": {
                    "search_volume": d["search_volume"],
                    "competition": round((d["difficulty"] % 100) / 100, 2),
                    "cpc": round(1 + (d["difficulty"] % 40) / 10, 2),
                },
                "keyword_properties": {
                    "keyword_difficulty": d["difficulty"],
                    "detected_language": result["language_code"],
                },
            })
        result["items"] = items
        result["items_count"] = len(items)
        result["total_count"] = len(items)
        return env

    def _chatgpt(self, task: dict[str, Any]) -> dict[str, Any]:
        prompt = task.get("user_prompt", "")
        d = _derive(prompt)
        env = _load("chatgpt_envelope")
        result = env["tasks"][0]["result"][0]
        result["model_name"] = task.get("model_name", "gpt-4o-mini")
        result["web_search"] = bool(task.get("web_search", False))
        result["input_tokens"] = max(1, len(prompt) // 4)
        result["output_tokens"] = 180

        mentioned = list(self.pool[:3])
        if self.domain_hint and d["chatgpt_mentions"]:
            mentioned.insert(1, self.domain_hint)
        text = (
            "Based on current coverage, the tools most often recommended here are "
            + ", ".join(mentioned)
            + ". Each differs in pricing and collaboration depth."
        )
        result["items"] = [{
            "type": "message",
            "sections": [{"type": "text", "text": text}],
        }]
        return env
