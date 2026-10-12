"""Unit tests for ProspectiveAgency orchestrator."""

from __future__ import annotations

import pytest

from organism.agency.checkpoint import PROSPECTIVE_AGENCY_SCHEMA_VERSION
from organism.agency.policy import ProspectivePolicy
from organism.agency.prospective import ProspectiveAgency
from organism.agency.types import (
    CounterfactualPrediction,
    ProspectiveCandidate,
)
from organism.agency.value import OutcomeValueLedger, context_key_for


def _make_agency(
    *,
    organism_id: str = "test-org",
    ledger: OutcomeValueLedger | None = None,
    min_model_confidence: float = 0.1,
    min_value_samples: int = 1,
    decision_margin: float = 0.0,
    query_budget: int = 8,
) -> ProspectiveAgency:
    if ledger is None:
        ledger = OutcomeValueLedger()
    policy = ProspectivePolicy(
        organism_id=organism_id,
        min_model_confidence=min_model_confidence,
        min_value_samples=min_value_samples,
        decision_margin=decision_margin,
    )
    return ProspectiveAgency(
        organism_id=organism_id,
        outcome_value_ledger=ledger,
        policy=policy,
        query_budget=query_budget,
    )


def _candidate(action_id: str, family: str = "primitive") -> ProspectiveCandidate:
    return ProspectiveCandidate(action_id=action_id, family=family)


def _predictor_always_positive(action_id: str, ctx):
    return CounterfactualPrediction(
        action_id=action_id,
        predicted_outcome="outcome.positive",
        confidence_class=6,
    )


def _predictor_raises(action_id: str, ctx):
    raise RuntimeError("model failure")


def test_dead_organism_returns_physiology_dead():
    agency = _make_agency()
    decision = agency.deliberate(
        tick=1,
        candidates=[_candidate("prim.a")],
        context_tokens=("ctx.1",),
        homeostatic_deviation=0.5,
        predictor=_predictor_always_positive,
        organism_alive=False,
    )
    assert decision.reason == "physiology_dead"
    assert decision.candidate_id is None


def test_no_active_model_returns_no_active_model():
    agency = _make_agency()
    decision = agency.deliberate(
        tick=1,
        candidates=[_candidate("prim.a")],
        context_tokens=("ctx.1",),
        homeostatic_deviation=0.5,
        predictor=_predictor_always_positive,
        has_active_model=False,
    )
    assert decision.reason == "no_active_model"


def test_no_candidates_returns_no_candidates_or_similar():
    agency = _make_agency()
    decision = agency.deliberate(
        tick=1,
        candidates=[],
        context_tokens=("ctx.1",),
        homeostatic_deviation=0.5,
        predictor=_predictor_always_positive,
    )
    assert decision.reason in {"no_candidates", "no_value_evidence"}


def test_predictor_exception_treated_as_no_candidate():
    ledger = OutcomeValueLedger()
    ledger.observe("outcome.positive", 0.5, tick=0)
    agency = _make_agency(ledger=ledger)
    decision = agency.deliberate(
        tick=1,
        candidates=[_candidate("prim.fail")],
        context_tokens=("ctx.1",),
        homeostatic_deviation=0.5,
        predictor=_predictor_raises,
    )
    assert decision.reason in {"no_candidates", "no_value_evidence", "insufficient_confidence"}


def test_successful_selection_with_sufficient_evidence():
    ledger = OutcomeValueLedger()
    for tick in range(8):
        ledger.observe("outcome.positive", 0.7, tick=tick)

    agency = _make_agency(ledger=ledger)
    decision = agency.deliberate(
        tick=10,
        candidates=[_candidate("prim.alpha")],
        context_tokens=("ctx.a", "ctx.b"),
        homeostatic_deviation=0.3,
        predictor=_predictor_always_positive,
    )
    assert decision.reason == "selected"
    assert decision.candidate_id == "prim.alpha"
    assert decision.predicted_outcome == "outcome.positive"
    assert decision.expected_value is not None
    assert decision.tick == 10
    assert decision.counterfactual_prediction_ref is not None
    assert decision.counterfactual_prediction_ref.startswith("model-prediction.")


def test_supplied_counterfactual_prediction_identity_is_preserved():
    ledger = OutcomeValueLedger()
    for tick in range(8):
        ledger.observe("outcome.positive", 0.7, tick=tick)
    agency = _make_agency(ledger=ledger)

    decision = agency.deliberate(
        tick=12,
        candidates=[_candidate("prim.alpha")],
        context_tokens=("ctx.a",),
        homeostatic_deviation=0.5,
        predictor=lambda aid, ctx: CounterfactualPrediction(
            action_id=aid,
            predicted_outcome="outcome.positive",
            confidence_class=7,
            prediction_ref="model-prediction.explicit",
        ),
    )

    assert decision.reason == "selected"
    assert decision.counterfactual_prediction_ref == "model-prediction.explicit"


def test_same_predicted_outcome_can_be_selected_or_rejected_by_private_context():
    ledger = OutcomeValueLedger()
    low_context = ("sense.rec.103", "state.sense.x.level.1")
    high_context = ("sense.rec.103", "state.sense.x.level.6")
    low_key = context_key_for(low_context)
    high_key = context_key_for(high_context)
    for tick in range(8):
        ledger.observe("outcome.same", 0.7, tick=tick, context_key=low_key)
        ledger.observe("outcome.same", -0.7, tick=tick, context_key=high_key)

    agency = _make_agency(ledger=ledger)
    predictor = lambda aid, ctx: CounterfactualPrediction(
        action_id=aid,
        predicted_outcome="outcome.same",
        confidence_class=7,
    )

    selected = agency.deliberate(
        tick=20,
        candidates=[_candidate("prim.same")],
        context_tokens=low_context,
        homeostatic_deviation=0.8,
        predictor=predictor,
    )
    rejected = agency.deliberate(
        tick=21,
        candidates=[_candidate("prim.same")],
        context_tokens=high_context,
        homeostatic_deviation=0.8,
        predictor=predictor,
    )

    assert selected.reason == "selected"
    assert selected.candidate_id == "prim.same"
    assert rejected.reason == "insufficient_margin"
    assert rejected.candidate_id is None


def test_query_budget_limits_candidates_queried():
    query_calls = []

    def _counting_predictor(action_id, ctx):
        query_calls.append(action_id)
        return CounterfactualPrediction(
            action_id=action_id,
            predicted_outcome=f"outcome.{action_id}",
            confidence_class=5,
        )

    agency = _make_agency(query_budget=3)
    candidates = [_candidate(f"prim.{i}") for i in range(10)]
    agency.deliberate(
        tick=1,
        candidates=candidates,
        context_tokens=("ctx.1",),
        homeostatic_deviation=0.5,
        predictor=_counting_predictor,
    )
    assert len(query_calls) <= 3


def test_checkpoint_roundtrip_preserves_ledger():
    ledger = OutcomeValueLedger()
    context = ("ctx.checkpoint",)
    context_key = context_key_for(context)
    for tick in range(6):
        ledger.observe("outcome.checkpoint", 0.4, tick=tick, context_key=context_key)

    agency = _make_agency(ledger=ledger)
    checkpoint = agency.checkpoint()
    assert checkpoint["schema_version"] == PROSPECTIVE_AGENCY_SCHEMA_VERSION

    policy = ProspectivePolicy(
        organism_id="test-org",
        min_model_confidence=0.1,
        min_value_samples=1,
        decision_margin=0.0,
    )
    restored = ProspectiveAgency.restore(checkpoint, organism_id="test-org", policy=policy)

    original_est = agency.outcome_value_ledger.estimate(
        "outcome.checkpoint", context_key=context_key
    )
    restored_est = restored.outcome_value_ledger.estimate(
        "outcome.checkpoint", context_key=context_key
    )

    assert original_est is not None
    assert restored_est is not None
    assert abs(original_est.mean_value - restored_est.mean_value) < 1e-9


def test_checkpoint_roundtrip_preserves_query_budget():
    ledger = OutcomeValueLedger()
    ledger.observe("outcome.any", 0.5, tick=0)
    agency = _make_agency(ledger=ledger, query_budget=2)
    policy = ProspectivePolicy(
        organism_id="test-org",
        min_model_confidence=0.1,
        min_value_samples=1,
        decision_margin=0.0,
    )
    restored = ProspectiveAgency.restore(
        agency.checkpoint(), organism_id="test-org", policy=policy
    )
    queried: list[str] = []

    def predictor(action_id, context):
        queried.append(action_id)
        return CounterfactualPrediction(
            action_id=action_id,
            predicted_outcome="outcome.any",
            confidence_class=5,
        )

    restored.deliberate(
        tick=1,
        candidates=[_candidate(f"primitive.{index}") for index in range(10)],
        context_tokens=("context.one",),
        homeostatic_deviation=0.5,
        predictor=predictor,
    )
    assert len(queried) == 2


def test_agency_rejects_policy_with_different_organism_identity():
    policy = ProspectivePolicy(
        organism_id="policy-org",
        min_model_confidence=0.1,
        min_value_samples=1,
        decision_margin=0.0,
    )
    with pytest.raises(ValueError, match="must match agency organism_id"):
        ProspectiveAgency(
            organism_id="agency-org",
            outcome_value_ledger=OutcomeValueLedger(),
            policy=policy,
        )


def test_restore_rejects_checkpoint_with_different_organism_identity():
    agency = _make_agency(organism_id="source-org")
    checkpoint = agency.checkpoint()
    checkpoint["organism_id"] = "other-org"
    policy = ProspectivePolicy(
        organism_id="source-org",
        min_model_confidence=0.1,
        min_value_samples=1,
        decision_margin=0.0,
    )
    with pytest.raises(ValueError, match="organism_id conflicts with checkpoint"):
        ProspectiveAgency.restore(
            checkpoint,
            organism_id="source-org",
            policy=policy,
        )


def test_restore_rejects_different_policy_configuration():
    agency = _make_agency()
    incompatible_policy = ProspectivePolicy(
        organism_id="test-org",
        min_model_confidence=0.2,
        min_value_samples=1,
        decision_margin=0.0,
    )
    with pytest.raises(ValueError, match="policy configuration"):
        ProspectiveAgency.restore(
            agency.checkpoint(), organism_id="test-org", policy=incompatible_policy
        )


def test_restore_rejects_query_budget_override_conflict():
    agency = _make_agency(query_budget=3)
    policy = ProspectivePolicy(
        organism_id="test-org",
        min_model_confidence=0.1,
        min_value_samples=1,
        decision_margin=0.0,
    )
    with pytest.raises(ValueError, match="query_budget conflicts"):
        ProspectiveAgency.restore(
            agency.checkpoint(),
            organism_id="test-org",
            policy=policy,
            query_budget=4,
        )


def test_restore_fails_closed_on_wrong_schema():
    with pytest.raises(ValueError, match="schema"):
        policy = ProspectivePolicy(
            organism_id="test-org",
            min_model_confidence=0.1,
            min_value_samples=1,
            decision_margin=0.0,
        )
        ProspectiveAgency.restore(
            {"schema_version": 99},
            organism_id="test-org",
            policy=policy,
        )


def test_restore_rejects_previous_schema_without_configuration():
    policy = ProspectivePolicy(
        organism_id="test-org",
        min_model_confidence=0.1,
        min_value_samples=1,
        decision_margin=0.0,
    )
    with pytest.raises(ValueError, match="schema"):
        ProspectiveAgency.restore(
            {"schema_version": 1, "outcome_value_ledger": {}},
            organism_id="test-org",
            policy=policy,
        )


def test_restore_requires_emitted_checkpoint_fields_and_rejects_unknown_fields():
    policy = ProspectivePolicy(
        organism_id="test-org",
        min_model_confidence=0.1,
        min_value_samples=1,
        decision_margin=0.0,
    )
    checkpoint = _make_agency().checkpoint()
    missing = dict(checkpoint)
    del missing["outcome_value_ledger"]
    with pytest.raises(ValueError):
        ProspectiveAgency.restore(missing, organism_id="test-org", policy=policy)
    extra = {**checkpoint, "unexpected": True}
    with pytest.raises(ValueError):
        ProspectiveAgency.restore(extra, organism_id="test-org", policy=policy)


def test_deliberate_on_death_then_no_active_model():
    agency = _make_agency()
    decision = agency.deliberate(
        tick=5,
        candidates=[_candidate("prim.x")],
        context_tokens=("ctx.x",),
        homeostatic_deviation=0.8,
        predictor=_predictor_always_positive,
        organism_alive=False,
        has_active_model=False,
    )
    assert decision.reason == "physiology_dead"


def test_outcome_value_ledger_property():
    agency = _make_agency()
    assert isinstance(agency.outcome_value_ledger, OutcomeValueLedger)


def test_deliberate_returns_tick_in_decision():
    ledger = OutcomeValueLedger()
    for tick in range(4):
        ledger.observe("outcome.x", 0.5, tick=tick)
    agency = _make_agency(ledger=ledger)
    decision = agency.deliberate(
        tick=42,
        candidates=[_candidate("prim.x")],
        context_tokens=("ctx.1",),
        homeostatic_deviation=0.2,
        predictor=lambda aid, ctx: CounterfactualPrediction(
            action_id=aid, predicted_outcome="outcome.x", confidence_class=6
        ),
    )
    assert decision.tick == 42
