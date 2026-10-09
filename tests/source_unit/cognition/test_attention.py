from __future__ import annotations

import pytest

from organism.cognition.attention import AttentionBudget, AttentionCandidate


def test_candidate_validation_preserves_infinite_unknown_uncertainty():
    with pytest.raises(ValueError):
        AttentionCandidate(name="bad", uncertainty=float("nan"), cost=1.0)
    assert AttentionCandidate(name="unknown", uncertainty=float("inf"), cost=1.0)


@pytest.mark.parametrize("field", ["cost", "rank_cost"])
def test_candidate_rejects_non_finite_ranking_costs(field):
    kwargs = {"name": "a", "uncertainty": 1.0, "cost": 1.0, field: float("inf")}
    with pytest.raises(ValueError):
        AttentionCandidate(**kwargs)


def test_budget_requires_a_finite_positive_value():
    for value in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            AttentionBudget(budget=value)


def test_allocates_by_uncertainty_per_cost_then_fits_budget():
    candidates = [
        AttentionCandidate(name="a", uncertainty=4.0, cost=1.0),
        AttentionCandidate(name="b", uncertainty=2.0, cost=1.0),
        AttentionCandidate(name="c", uncertainty=1.0, cost=1.0),
    ]
    assert [item.name for item in AttentionBudget(budget=2.0).allocate(candidates)] == ["a", "b"]


def test_unknown_uncertainty_wins_and_ties_are_deterministic():
    budget = AttentionBudget(budget=1.0)
    candidates = [
        AttentionCandidate(name="z", uncertainty=1.0, cost=1.0),
        AttentionCandidate(name="unknown", uncertainty=float("inf"), cost=1.0),
        AttentionCandidate(name="a", uncertainty=1.0, cost=1.0),
    ]
    assert budget.allocate(candidates)[0].name == "unknown"
    tied = [candidates[0], candidates[2]]
    assert budget.allocate(tied)[0].name == "a"


def test_rank_cost_reorders_without_changing_consumed_cost():
    candidates = [
        AttentionCandidate(name="ranked-lower", uncertainty=10.0, cost=0.7, rank_cost=4.0),
        AttentionCandidate(name="ranked-first", uncertainty=10.0, cost=0.7),
    ]
    selected = AttentionBudget(budget=0.7).allocate(candidates)
    assert [(item.name, item.cost) for item in selected] == [("ranked-first", 0.7)]


def test_observation_count_applies_diminishing_return():
    candidates = [
        AttentionCandidate(name="repeated", uncertainty=2.0, cost=1.0, observations=100),
        AttentionCandidate(name="fresh", uncertainty=1.0, cost=1.0),
    ]
    assert AttentionBudget(budget=1.0).allocate(candidates)[0].name == "fresh"


def test_candidate_over_budget_is_not_selected():
    candidate = AttentionCandidate(name="a", uncertainty=1.0, cost=1.0)
    assert AttentionBudget(budget=0.5).allocate([candidate]) == ()
