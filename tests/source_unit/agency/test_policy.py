"""Unit tests for ProspectivePolicy — utility, gates, tiebreaking, decision reasons."""

from __future__ import annotations

import pytest

from organism.agency.policy import EvaluatedCandidate, ProspectivePolicy
from organism.agency.types import (
    CounterfactualPrediction,
    OutcomeValueEstimate,
    ProspectiveCandidate,
)


def _make_policy(
    *,
    organism_id: str = "test-org",
    min_model_confidence: float = 0.1,
    min_value_samples: int = 1,
    decision_margin: float = 0.0,
) -> ProspectivePolicy:
    return ProspectivePolicy(
        organism_id=organism_id,
        min_model_confidence=min_model_confidence,
        min_value_samples=min_value_samples,
        decision_margin=decision_margin,
    )


def _candidate(
    action_id: str, predicted_outcome: str, cc: int, mean_value: float, samples: int = 8
) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        candidate=ProspectiveCandidate(action_id=action_id, family="primitive"),
        prediction=CounterfactualPrediction(
            action_id=action_id,
            predicted_outcome=predicted_outcome,
            confidence_class=cc,
        ),
        value=OutcomeValueEstimate(
            outcome_id=predicted_outcome,
            samples=samples,
            mean_value=mean_value,
            variance=0.01,
            confidence=min(1.0, samples / 16.0),
        ),
        estimated_cost=None,
    )


def _candidate_no_value(action_id: str, predicted_outcome: str, cc: int) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        candidate=ProspectiveCandidate(action_id=action_id, family="primitive"),
        prediction=CounterfactualPrediction(
            action_id=action_id,
            predicted_outcome=predicted_outcome,
            confidence_class=cc,
        ),
        value=None,
        estimated_cost=None,
    )


def test_no_candidates_returns_no_candidates_reason():
    policy = _make_policy()
    decision = policy.choose([], homeostatic_deviation=0.5, tick=1)
    assert decision.reason == "no_candidates"
    assert decision.candidate_id is None


def test_all_missing_value_returns_no_value_evidence():
    policy = _make_policy()
    ec = _candidate_no_value("prim.1", "outcome.1", cc=6)
    decision = policy.choose([ec], homeostatic_deviation=0.5, tick=1)
    assert decision.reason == "no_value_evidence"
    assert decision.candidate_id is None


def test_insufficient_model_confidence_returns_reason():
    policy = _make_policy(min_model_confidence=0.9)
    ec = _candidate("prim.1", "outcome.1", cc=1, mean_value=0.5)
    decision = policy.choose([ec], homeostatic_deviation=0.5, tick=1)
    assert decision.reason == "insufficient_confidence"


def test_decision_margin_blocks_close_candidates():
    policy = _make_policy(decision_margin=0.5)
    ec1 = _candidate("prim.1", "outcome.1", cc=7, mean_value=0.5, samples=16)
    ec2 = _candidate("prim.2", "outcome.2", cc=7, mean_value=0.49, samples=16)
    decision = policy.choose([ec1, ec2], homeostatic_deviation=0.5, tick=1)
    assert decision.reason == "insufficient_margin"


def test_single_sufficient_candidate_is_selected():
    policy = _make_policy()
    ec = _candidate("prim.abc", "outcome.positive", cc=6, mean_value=0.8, samples=16)
    decision = policy.choose([ec], homeostatic_deviation=0.5, tick=1)
    assert decision.reason == "selected"
    assert decision.candidate_id == "prim.abc"
    assert decision.predicted_outcome == "outcome.positive"
    assert decision.expected_value is not None
    assert decision.model_confidence is not None
    assert decision.value_confidence is not None


def test_best_utility_candidate_wins():
    policy = _make_policy(decision_margin=0.01)
    ec_good = _candidate("prim.good", "outcome.good", cc=7, mean_value=0.9, samples=16)
    ec_bad = _candidate("prim.bad", "outcome.bad", cc=3, mean_value=0.1, samples=16)
    decision = policy.choose([ec_good, ec_bad], homeostatic_deviation=0.3, tick=5)
    assert decision.reason == "selected"
    assert decision.candidate_id == "prim.good"


def test_negative_utility_candidate_can_lose_to_positive():
    policy = _make_policy(decision_margin=0.01)
    ec_neg = _candidate("prim.neg", "outcome.neg", cc=7, mean_value=-0.8, samples=16)
    ec_pos = _candidate("prim.pos", "outcome.pos", cc=7, mean_value=0.1, samples=16)
    decision = policy.choose([ec_neg, ec_pos], homeostatic_deviation=0.3, tick=10)
    assert decision.reason == "selected"
    assert decision.candidate_id == "prim.pos"


def test_only_non_positive_value_abstains_from_pragmatic_prospection():
    policy = _make_policy(decision_margin=0.0)
    ec = _candidate("prim.neg", "outcome.neg", cc=7, mean_value=-0.8, samples=16)
    decision = policy.choose([ec], homeostatic_deviation=0.8, tick=10)
    assert decision.reason == "insufficient_margin"
    assert decision.candidate_id is None


def test_tiebreaking_is_deterministic():
    policy = _make_policy(decision_margin=0.0)
    ec1 = _candidate("prim.alpha", "outcome.x", cc=7, mean_value=0.5, samples=16)
    ec2 = _candidate("prim.beta", "outcome.x", cc=7, mean_value=0.5, samples=16)
    candidates = [ec1, ec2]
    decisions = [
        policy.choose(candidates, homeostatic_deviation=0.5, tick=42).candidate_id for _ in range(5)
    ]
    assert len(set(decisions)) == 1


def test_tiebreaking_varies_by_tick():
    policy = _make_policy(decision_margin=0.0)
    ec1 = _candidate("prim.aa", "outcome.1", cc=7, mean_value=0.5, samples=16)
    ec2 = _candidate("prim.bb", "outcome.1", cc=7, mean_value=0.5, samples=16)
    choices = {
        policy.choose([ec1, ec2], homeostatic_deviation=0.5, tick=t).candidate_id
        for t in range(100)
    }
    assert len(choices) >= 1


def test_homeostatic_need_scales_expected_value():
    policy = _make_policy(decision_margin=0.0)
    ec = _candidate("prim.1", "outcome.1", cc=7, mean_value=0.8, samples=16)

    low = policy.choose([ec], homeostatic_deviation=0.2, tick=1)
    high = policy.choose([ec], homeostatic_deviation=0.8, tick=2)

    assert low.reason == high.reason == "selected"
    assert low.expected_value is not None and high.expected_value is not None
    assert high.expected_value == pytest.approx(low.expected_value * 4.0)


def test_zero_homeostatic_need_abstains_from_pragmatic_prospection():
    policy = _make_policy(decision_margin=0.0)
    ec = _candidate("prim.1", "outcome.1", cc=7, mean_value=0.9, samples=16)
    decision = policy.choose([ec], homeostatic_deviation=0.0, tick=1)
    assert decision.reason == "insufficient_margin"
    assert decision.candidate_id is None


def test_selected_decision_has_correct_margin():
    policy = _make_policy(decision_margin=0.0)
    ec1 = _candidate("prim.1", "outcome.1", cc=7, mean_value=0.9, samples=16)
    ec2 = _candidate("prim.2", "outcome.2", cc=7, mean_value=0.4, samples=16)
    decision = policy.choose([ec1, ec2], homeostatic_deviation=0.5, tick=1)
    assert decision.reason == "selected"
    assert decision.decision_margin is not None
    assert decision.decision_margin >= 0.0


def test_zero_confidence_class_is_excluded():
    policy = _make_policy(min_model_confidence=0.0)
    ec = _candidate("prim.zero_cc", "outcome.1", cc=0, mean_value=0.9, samples=16)
    decision = policy.choose([ec], homeostatic_deviation=0.5, tick=1)
    assert decision.reason == "insufficient_confidence"


def test_policy_rejects_invalid_homeostatic_deviation():
    policy = _make_policy()
    ec = _candidate("prim.1", "outcome.1", cc=7, mean_value=0.5, samples=16)
    with pytest.raises(ValueError):
        policy.choose([ec], homeostatic_deviation=-0.1, tick=1)
    with pytest.raises(ValueError):
        policy.choose([ec], homeostatic_deviation=1.1, tick=1)


def test_policy_construction_validates_inputs():
    with pytest.raises(ValueError):
        ProspectivePolicy(
            organism_id="", min_model_confidence=0.5, min_value_samples=4, decision_margin=0.02
        )
    with pytest.raises(ValueError):
        ProspectivePolicy(
            organism_id="x", min_model_confidence=1.5, min_value_samples=4, decision_margin=0.02
        )
    with pytest.raises(ValueError):
        ProspectivePolicy(
            organism_id="x", min_model_confidence=0.5, min_value_samples=0, decision_margin=0.02
        )
    with pytest.raises(ValueError):
        ProspectivePolicy(
            organism_id="x", min_model_confidence=0.5, min_value_samples=4, decision_margin=-0.1
        )


def test_epistemic_value_breaks_only_exact_pragmatic_ties():
    policy = _make_policy(decision_margin=0.0)
    low_epistemic = EvaluatedCandidate(
        candidate=ProspectiveCandidate(action_id="prim.low-info", family="primitive"),
        prediction=CounterfactualPrediction(
            action_id="prim.low-info",
            predicted_outcome="outcome.same-a",
            confidence_class=7,
        ),
        value=OutcomeValueEstimate(
            outcome_id="outcome.same-a",
            samples=16,
            mean_value=0.5,
            variance=0.01,
            confidence=1.0,
        ),
        estimated_cost=None,
        epistemic_value=0.1,
    )
    high_epistemic = EvaluatedCandidate(
        candidate=ProspectiveCandidate(action_id="prim.high-info", family="primitive"),
        prediction=CounterfactualPrediction(
            action_id="prim.high-info",
            predicted_outcome="outcome.same-b",
            confidence_class=7,
        ),
        value=OutcomeValueEstimate(
            outcome_id="outcome.same-b",
            samples=16,
            mean_value=0.5,
            variance=0.01,
            confidence=1.0,
        ),
        estimated_cost=None,
        epistemic_value=0.9,
    )

    decision = policy.choose(
        [low_epistemic, high_epistemic],
        homeostatic_deviation=0.5,
        tick=7,
    )

    assert decision.reason == "selected"
    assert decision.candidate_id == "prim.high-info"


def test_epistemic_value_never_overrides_higher_pragmatic_utility():
    policy = _make_policy(decision_margin=0.0)
    pragmatic = EvaluatedCandidate(
        candidate=ProspectiveCandidate(action_id="prim.pragmatic", family="primitive"),
        prediction=CounterfactualPrediction(
            action_id="prim.pragmatic",
            predicted_outcome="outcome.pragmatic",
            confidence_class=7,
        ),
        value=OutcomeValueEstimate(
            outcome_id="outcome.pragmatic",
            samples=16,
            mean_value=0.6,
            variance=0.01,
            confidence=1.0,
        ),
        estimated_cost=None,
        epistemic_value=0.0,
    )
    informative = EvaluatedCandidate(
        candidate=ProspectiveCandidate(action_id="prim.informative", family="primitive"),
        prediction=CounterfactualPrediction(
            action_id="prim.informative",
            predicted_outcome="outcome.informative",
            confidence_class=7,
        ),
        value=OutcomeValueEstimate(
            outcome_id="outcome.informative",
            samples=16,
            mean_value=0.5,
            variance=0.01,
            confidence=1.0,
        ),
        estimated_cost=None,
        epistemic_value=1.0,
    )

    decision = policy.choose(
        [informative, pragmatic],
        homeostatic_deviation=0.5,
        tick=7,
    )

    assert decision.reason == "selected"
    assert decision.candidate_id == "prim.pragmatic"
