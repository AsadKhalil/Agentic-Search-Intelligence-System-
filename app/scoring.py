"""Deterministic opportunity score (PLAN §7). Range [0.02, 1.0].

Kept out of the LLM's hands on purpose: a score that drifts between runs is not a metric.
"""
from math import log10

VOLUME_CEILING = 10_000
_LOG_CEILING = log10(1 + VOLUME_CEILING)

W_VOLUME, W_DIFFICULTY, W_VISIBILITY = 0.45, 0.35, 0.20


def volume_norm(volume: int | None) -> float:
    if volume is None or volume <= 0:
        return 0.0
    return min(1.0, log10(1 + volume) / _LOG_CEILING)


def difficulty_inv(difficulty: float | None) -> float:
    if difficulty is None:
        return 0.5
    return 1 - min(100.0, max(0.0, float(difficulty))) / 100


def visibility_gap(domain_visible: bool | None, position: int | None) -> float:
    """Organic visibility only. Unknown leans mildly toward opportunity, never 0 or 1."""
    if domain_visible is None:
        return 0.60
    if domain_visible is False:
        return 1.00
    if position is None:
        return 0.50
    if position <= 3:
        return 0.10
    if position <= 5:
        return 0.35
    return 0.50


def opportunity_score(
    *,
    search_volume: int | None,
    competitive_difficulty: float | None,
    domain_visible: bool | None,
    visibility_position: int | None,
) -> float:
    return round(
        W_VOLUME * volume_norm(search_volume)
        + W_DIFFICULTY * difficulty_inv(competitive_difficulty)
        + W_VISIBILITY * visibility_gap(domain_visible, visibility_position),
        4,
    )


def apply_scores(merged: list) -> list:
    """Score every merged query in place. Shared by analyze and fallback_analysis so the
    number never depends on which path produced the prose."""
    for row in merged:
        row.opportunity_score = opportunity_score(
            search_volume=row.search_volume,
            competitive_difficulty=row.competitive_difficulty,
            domain_visible=row.domain_visible,
            visibility_position=row.visibility_position,
        )
    return merged
