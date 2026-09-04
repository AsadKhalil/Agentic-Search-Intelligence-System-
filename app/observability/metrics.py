"""Per-run metric aggregation, built from the NodeEvent list the graph accumulates."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a cycle: schemas imports nothing from observability
    from app.schemas import NodeEvent


class RunMetrics:
    """Aggregates node events into the shape returned by /run and stored on the run row."""

    def __init__(self, events: list["NodeEvent"]) -> None:
        self.events = events

    def as_dict(self) -> dict[str, Any]:
        per_node: dict[str, dict[str, Any]] = {}
        by_name: dict[str, list["NodeEvent"]] = defaultdict(list)
        for event in self.events:
            by_name[event.node].append(event)

        for name, events in by_name.items():
            durations = [e.duration_ms for e in events]
            per_node[name] = {
                "invocations": len(events),
                "duration_ms_total": round(sum(durations), 2),
                "duration_ms_max": round(max(durations), 2),
                "succeeded": sum(1 for e in events if e.ok),
                "failed": sum(1 for e in events if not e.ok),
                "retries": sum(e.retries for e in events),
                "api_calls": sum(e.api_calls for e in events),
            }

        return {
            "nodes": per_node,
            "node_sequence": [e.node for e in self.events],
            "total_duration_ms": round(sum(e.duration_ms for e in self.events), 2),
            "total_api_calls": sum(e.api_calls for e in self.events),
            "total_retries": sum(e.retries for e in self.events),
            "total_tokens": sum(e.tokens for e in self.events),
            "failed_nodes": [e.node for e in self.events if not e.ok],
        }

    @property
    def tokens_used(self) -> int:
        return sum(e.tokens for e in self.events)
