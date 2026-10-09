"""Bounded attention-allocation values and heuristic.

This module accepts caller-provided candidate names, uncertainty and costs. It
does not observe sensors, infer threats, inspect outcomes, or execute a choice.
The allocation policy is the source implementation's greedy uncertainty per
cost heuristic, not an optimal knapsack solver or a scientific claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True, frozen=True)
class AttentionCandidate:
    """One candidate for a bounded attention allocation."""

    name: str
    uncertainty: float
    cost: float
    rank_cost: float = 1.0
    observations: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.cost, bool) or not math.isfinite(self.cost) or self.cost <= 0.0:
            raise ValueError("cost must be positive")
        if (
            isinstance(self.rank_cost, bool)
            or not math.isfinite(self.rank_cost)
            or self.rank_cost <= 0.0
        ):
            raise ValueError("rank_cost must be positive")
        # +inf is allowed for an explicitly unknown uncertainty; NaN and
        # negative values do not provide a deterministic ranking.
        if (
            isinstance(self.uncertainty, bool)
            or math.isnan(self.uncertainty)
            or self.uncertainty < 0.0
        ):
            raise ValueError("uncertainty must be non-negative")
        if (
            isinstance(self.observations, bool)
            or not isinstance(self.observations, int)
            or self.observations < 0
        ):
            raise ValueError("observations must be non-negative")


@dataclass(slots=True, frozen=True)
class AttentionAllocation:
    """One candidate selected by the bounded allocator."""

    name: str
    uncertainty: float
    cost: float


class AttentionBudget:
    """Greedily allocate a positive budget by uncertainty per ranking cost.

    This is O(n log n), deterministic (ties sort by name), and not an exact
    0/1 knapsack solver. Candidate costs determine budget consumption;
    ``rank_cost`` changes ranking only. Repeated observations apply the
    source heuristic's diminishing-return factor.
    """

    def __init__(self, *, budget: float) -> None:
        if isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("budget must be positive")
        self._budget = budget

    @property
    def budget(self) -> float:
        return self._budget

    def allocate(self, candidates: Iterable[AttentionCandidate]) -> tuple[AttentionAllocation, ...]:
        def score(candidate: AttentionCandidate) -> float:
            diminishing = 1.0 / (1.0 + 0.05 * candidate.observations)
            return (candidate.uncertainty * diminishing) / (candidate.cost * candidate.rank_cost)

        ranked = sorted(candidates, key=lambda candidate: (-score(candidate), candidate.name))
        selected: list[AttentionAllocation] = []
        remaining = self._budget
        for candidate in ranked:
            if candidate.cost <= remaining:
                selected.append(
                    AttentionAllocation(
                        name=candidate.name,
                        uncertainty=candidate.uncertainty,
                        cost=candidate.cost,
                    )
                )
                remaining -= candidate.cost
        return tuple(selected)


__all__ = ["AttentionAllocation", "AttentionBudget", "AttentionCandidate"]
