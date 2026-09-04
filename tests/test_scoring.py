"""Opportunity-score boundary table (PLAN §7, §9.8)."""
import pytest

from app.scoring import difficulty_inv, opportunity_score, visibility_gap, volume_norm


def score(volume=1000, difficulty=50, visible=False, position=None):
    return opportunity_score(search_volume=volume, competitive_difficulty=difficulty,
                             domain_visible=visible, visibility_position=position)


@pytest.mark.parametrize("visible,position,expected", [
    (None, None, 0.60),    # unknown: a mild lean, never a confident 0 or 1
    (None, 2, 0.60),       # position is meaningless when visibility is unknown
    (False, None, 1.00),   # absent = maximum opportunity
    (True, 1, 0.10),
    (True, 3, 0.10),
    (True, 4, 0.35),
    (True, 5, 0.35),
    (True, 6, 0.50),
    (True, 50, 0.50),
    (True, None, 0.50),    # visible but position unknown
])
def test_visibility_gap_boundaries(visible, position, expected):
    assert visibility_gap(visible, position) == expected


@pytest.mark.parametrize("volume,expected", [
    (None, 0.0), (0, 0.0), (-5, 0.0),
    (10_000, 1.0), (50_000, 1.0),   # clamped at the ceiling
])
def test_volume_norm_edges(volume, expected):
    assert volume_norm(volume) == expected


def test_volume_norm_is_monotonic():
    values = [volume_norm(v) for v in (10, 100, 1_000, 5_000, 10_000)]
    assert values == sorted(values)


@pytest.mark.parametrize("difficulty,expected", [
    (None, 0.5),    # unknown difficulty sits in the middle
    (0, 1.0), (50, 0.5), (100, 0.0),
    (150, 0.0), (-20, 1.0),   # clamped
])
def test_difficulty_inv_edges(difficulty, expected):
    assert difficulty_inv(difficulty) == expected


def test_score_bounds():
    best = score(volume=10_000, difficulty=0, visible=False)
    worst = score(volume=None, difficulty=100, visible=True, position=1)
    assert best == 1.0
    assert worst == 0.02
    assert 0.02 <= score(volume=None, difficulty=None, visible=None) <= 1.0


def test_position_four_scores_higher_than_position_three():
    """Rank 4 is off the fold, so it carries more remaining opportunity than rank 3."""
    assert score(visible=True, position=4) > score(visible=True, position=3)
    assert score(visible=True, position=6) > score(visible=True, position=5)


def test_unknown_sits_between_visible_and_absent():
    absent = score(visible=False)
    unknown = score(visible=None)
    top_three = score(visible=True, position=2)
    assert top_three < unknown < absent


def test_nulls_do_not_produce_zero():
    """A query with no metrics at all still scores, so failed rows remain rankable."""
    assert score(volume=None, difficulty=None, visible=None) == pytest.approx(0.295)
