"""Prospective decision policy for L8 Prospective Agency.

Pure decision logic. Receives pre-evaluated candidates and returns a
DeliberationOutcome. Knows nothing about physiology fields, motor execution, or
environment semantics.

Pragmatic utility is endogenous historical value gated by current constitutional
need:

    utility = mean_value
              × value_confidence
              × model_confidence_normalized
              × homeostatic_deviation

A non-positive pragmatic utility is never selected prospectively. Exploration
and epistemic admission remain available through the model-free executive path.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection
from dataclasses import dataclass

from .types import (
    CounterfactualPrediction,
    DeliberationOutcome,
    OutcomeValueEstimate,
    ProspectiveCandidate,
)

_MIN_CONFIDENCE_CLASS = 1
_CONFIDENCE_CLASS_MAX = 7


@dataclass(frozen=True)
class EvaluatedCandidate:
    """One candidate after model prediction and value lookup have been applied."""

    candidate: ProspectiveCandidate
    prediction: CounterfactualPrediction
    value: OutcomeValueEstimate | None
    estimated_cost: float | None
    epistemic_value: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.epistemic_value, bool)
            or not isinstance(self.epistemic_value, (int, float))
            or not math.isfinite(float(self.epistemic_value))
            or not 0.0 <= float(self.epistemic_value) <= 1.0
        ):
            raise ValueError("epistemic_value must be within [0, 1]")


class ProspectivePolicy:
    """Conservative selection policy for prospective action deliberation."""

    def __init__(
        self,
        *,
        organism_id: str,
        min_model_confidence: float = 0.5,
        min_value_samples: int = 4,
        decision_margin: float = 0.02,
    ) -> None:
        if not isinstance(organism_id, str) or not organism_id:
            raise ValueError("organism_id must be a non-empty string")
        if (
            isinstance(min_model_confidence, bool)
            or not isinstance(min_model_confidence, (int, float))
            or not math.isfinite(float(min_model_confidence))
            or not 0.0 <= float(min_model_confidence) <= 1.0
        ):
            raise ValueError("min_model_confidence must be in [0, 1]")
        if (
            isinstance(min_value_samples, bool)
            or not isinstance(min_value_samples, int)
            or min_value_samples < 1
        ):
            raise ValueError("min_value_samples must be a positive integer")
        if (
            isinstance(decision_margin, bool)
            or not isinstance(decision_margin, (int, float))
            or not math.isfinite(float(decision_margin))
            or float(decision_margin) < 0.0
        ):
            raise ValueError("decision_margin must be a non-negative finite float")

        self._organism_id = organism_id
        self._min_model_confidence = float(min_model_confidence)
        self._min_value_confidence = float(min_value_samples) / 16.0
        self._decision_margin = float(decision_margin)

    def _tiebreak_key(self, action_id: str, tick: int) -> str:
        material = f"{self._organism_id}:{tick}:{action_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    @staticmethod
    def _utility(
        value: OutcomeValueEstimate,
        confidence_class: int,
        *,
        homeostatic_deviation: float,
    ) -> float:
        model_conf = confidence_class / float(_CONFIDENCE_CLASS_MAX)
        urgency = max(0.0, min(1.0, float(homeostatic_deviation)))
        return float(value.mean_value) * float(value.confidence) * model_conf * urgency

    def choose(
        self,
        candidates: Collection[EvaluatedCandidate],
        *,
        homeostatic_deviation: float,
        tick: int,
    ) -> DeliberationOutcome:
        """Select the best positively-valued candidate, or abstain."""
        if (
            isinstance(homeostatic_deviation, bool)
            or not isinstance(homeostatic_deviation, (int, float))
            or not math.isfinite(float(homeostatic_deviation))
            or not 0.0 <= float(homeostatic_deviation) <= 1.0
        ):
            raise ValueError("homeostatic_deviation must be within [0, 1]")

        candidate_list = list(candidates)
        if not candidate_list:
            return DeliberationOutcome(
                tick=tick,
                candidate_id=None,
                predicted_outcome=None,
                expected_value=None,
                model_confidence=None,
                value_confidence=None,
                decision_margin=None,
                reason="no_candidates",
            )

        valued = [ec for ec in candidate_list if ec.value is not None]
        if not valued:
            return DeliberationOutcome(
                tick=tick,
                candidate_id=None,
                predicted_outcome=None,
                expected_value=None,
                model_confidence=None,
                value_confidence=None,
                decision_margin=None,
                reason="no_value_evidence",
            )

        confident: list[tuple[float, float, str, EvaluatedCandidate]] = []
        for ec in valued:
            assert ec.value is not None
            model_conf_norm = ec.prediction.confidence_class / float(_CONFIDENCE_CLASS_MAX)
            if model_conf_norm < self._min_model_confidence:
                continue
            if ec.value.confidence < self._min_value_confidence:
                continue
            if ec.prediction.confidence_class < _MIN_CONFIDENCE_CLASS:
                continue
            utility = self._utility(
                ec.value,
                ec.prediction.confidence_class,
                homeostatic_deviation=float(homeostatic_deviation),
            )
            tiebreak = self._tiebreak_key(ec.candidate.action_id, tick)
            confident.append((utility, float(ec.epistemic_value), tiebreak, ec))

        if not confident:
            return DeliberationOutcome(
                tick=tick,
                candidate_id=None,
                predicted_outcome=None,
                expected_value=None,
                model_confidence=None,
                value_confidence=None,
                decision_margin=None,
                reason="insufficient_confidence",
            )

        confident.sort(key=lambda item: (-item[0], -item[1], item[2]))
        best_utility, _, _, best_ec = confident[0]

        if best_utility <= 0.0:
            return DeliberationOutcome(
                tick=tick,
                candidate_id=None,
                predicted_outcome=None,
                expected_value=None,
                model_confidence=None,
                value_confidence=None,
                decision_margin=0.0,
                reason="insufficient_margin",
            )

        second_utility = confident[1][0] if len(confident) > 1 else -math.inf
        margin = best_utility - second_utility
        if margin < self._decision_margin:
            return DeliberationOutcome(
                tick=tick,
                candidate_id=None,
                predicted_outcome=None,
                expected_value=None,
                model_confidence=None,
                value_confidence=None,
                decision_margin=float(margin) if math.isfinite(margin) else None,
                reason="insufficient_margin",
            )

        assert best_ec.value is not None
        return DeliberationOutcome(
            tick=tick,
            candidate_id=best_ec.candidate.action_id,
            predicted_outcome=best_ec.prediction.predicted_outcome,
            expected_value=float(best_utility),
            model_confidence=best_ec.prediction.confidence_class / float(_CONFIDENCE_CLASS_MAX),
            value_confidence=float(best_ec.value.confidence),
            decision_margin=float(margin) if math.isfinite(margin) else None,
            reason="selected",
            counterfactual_prediction_ref=best_ec.prediction.prediction_ref,
        )


__all__ = [
    "EvaluatedCandidate",
    "ProspectivePolicy",
]
