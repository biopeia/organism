"""Unit and regression tests for evidence-grounded causal controllability confidence."""

import pytest

from organism.actuation.model import (
    CausalSourceKind,
    ControllabilityModel,
    controllability_confidence,
)


def test_zero_action_linked_hits_never_yield_positive_confidence():
    """Requirement 1: Positive confidence requires positive action-linked evidence."""
    assert controllability_confidence(8, 0, 0, 0) == 0.0
    assert controllability_confidence(8, 0, 8, 0) == 0.0
    assert controllability_confidence(8, 0, 8, 8) == 0.0


def test_no_action_opportunities_yield_zero_confidence():
    """Zero action opportunities cannot produce positive confidence."""
    assert controllability_confidence(0, 0, 8, 0) == 0.0
    assert controllability_confidence(0, 0, 0, 0) == 0.0
    assert controllability_confidence(-1, 0, 0, 0) == 0.0


def test_uncontrasted_baseline_scaled_by_reliability_and_support():
    """When other_n == 0, confidence scales with reliability and support up to 0.25 ceiling."""
    # 8/8 hits uncontrasted: reliability = 1.0, hits/8 = 1.0 -> 0.25
    assert controllability_confidence(8, 8, 0, 0) == 0.25

    # 4/8 hits uncontrasted: reliability = 0.5, hits/8 = 0.5 -> 0.5 * 0.25 * 0.5 = 0.0625
    assert controllability_confidence(8, 4, 0, 0) == pytest.approx(0.0625)

    # 2/2 hits uncontrasted: reliability = 1.0, hits/8 = 0.25 -> 1.0 * 0.25 * 0.25 = 0.0625
    assert controllability_confidence(2, 2, 0, 0) == pytest.approx(0.0625)

    # Zero hits uncontrasted is 0.0
    assert controllability_confidence(8, 0, 0, 0) == 0.0


def test_passive_baseline_contrast():
    """Contrasting against quiet passive baseline yields full attribution up to 1.0."""
    # 8/8 hits with 8 quiet passive windows (0/8 hits): advantage = 1.0 - 0.0 = 1.0, support = 1.0 -> 1.0
    assert controllability_confidence(8, 8, 8, 0) == 1.0

    # 8/8 hits with 4 quiet passive windows: advantage = 1.0, support = 4/8 = 0.5 -> 0.5
    assert controllability_confidence(8, 8, 4, 0) == 0.5


def test_competing_or_pervasive_alternative_reduces_or_erases_confidence():
    """When alternative rate equals or exceeds action rate, causal confidence is 0.0."""
    # Action 8/8, but alternative also 8/8 (e.g. effect occurs 100% of the time anyway)
    assert controllability_confidence(8, 8, 8, 8) == 0.0

    # Action 4/8 (50%), alternative 6/8 (75%) -> advantage < 0 -> 0.0
    assert controllability_confidence(8, 4, 8, 6) == 0.0


def test_monotonic_increase_with_evidence_support():
    """Confidence increases monotonically with accumulating supportive evidence."""
    # Uncontrasted evidence growth
    conf_1 = controllability_confidence(1, 1, 0, 0)
    conf_2 = controllability_confidence(2, 2, 0, 0)
    conf_4 = controllability_confidence(4, 4, 0, 0)
    conf_8 = controllability_confidence(8, 8, 0, 0)
    assert 0.0 < conf_1 < conf_2 < conf_4 < conf_8 == 0.25

    # Contrasted evidence growth against quiet baseline
    c_contrasted_2 = controllability_confidence(2, 2, 2, 0)
    c_contrasted_4 = controllability_confidence(4, 4, 4, 0)
    c_contrasted_8 = controllability_confidence(8, 8, 8, 0)
    assert 0.0 < c_contrasted_2 < c_contrasted_4 < c_contrasted_8 == 1.0


def test_partial_advantage_with_support():
    """Partial advantage when alternative rate is non-zero but below action rate."""
    # Action 8/8 (1.0), Alternative 2/8 (0.25). Advantage = 0.75, support = 1.0 -> 0.75
    assert controllability_confidence(8, 8, 8, 2) == pytest.approx(0.75)


def test_controllability_model_update_and_estimate():
    """ControllabilityModel correctly indexes and exposes ControllabilityEstimate."""
    class DummyOpportunityLedger:
        def effect_opportunities(self, effect_id, *, competence_id, context_ref):
            if competence_id == "comp.a":
                return (8, 8, 8, 0)
            return (0, 0, 0, 0)

    ledger = DummyOpportunityLedger()
    model = ControllabilityModel()
    estimate = model.update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.COMPETENCE,
        source_ref="comp.a",
        effect_id="effect.x",
        context_id=None,
        tick=10,
    )

    assert estimate.confidence == 1.0
    assert estimate.reliability == 1.0
    assert estimate.counterfactual_rate == 0.0
    assert estimate.causal_advantage == 1.0
    assert estimate.action_support == 8
    assert estimate.counterfactual_support == 8
    assert estimate.last_updated_tick == 10

    retrieved = model.estimate(
        source_kind=CausalSourceKind.COMPETENCE,
        source_ref="comp.a",
        effect_id="effect.x",
        context_id=None,
    )
    assert retrieved is estimate
