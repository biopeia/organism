"""Unit and regression tests for evidence-grounded causal controllability confidence."""

import pytest

from organism.actuation.model import (
    AgencyModel,
    CausalSourceKind,
    ControllabilityModel,
    controllability_confidence,
)
from organism.actuation.evidence import CausalEvidenceLedger, SensorimotorTransition


def _record_action(
    ledger: CausalEvidenceLedger,
    *,
    signature: str,
    commitment: str,
    effect: str | None,
    tick: int,
    competence: str | None = None,
    context: str | None = "context.matched",
) -> None:
    ledger.observe(
        SensorimotorTransition(
            transition_id=f"transition.{tick}",
            tick_start=tick,
            tick_end=tick + 1,
            context_ref=context,
            commitment_id=commitment,
            controller_id="controller.test",
            competence_id=competence,
            attempt_id=f"attempt.{tick}",
            intervention_signature_id=signature,
            state_before_ref=f"state.before.{tick}",
            motor_command_ref=f"command.{tick}",
            actuation_ref=f"actuation.{tick}",
            prediction_ref=None,
            state_after_ref=f"state.after.{tick}",
            observed_effect_id=effect,
        )
    )


def _record_passive(
    ledger: CausalEvidenceLedger,
    *,
    effect: str | None,
    tick: int,
    context: str | None = "context.matched",
) -> None:
    ledger.observe_passive_window(
        tick_start=tick,
        tick_end=tick + 1,
        context_ref=context,
        prior_state_ref=f"passive.before.{tick}",
        resulting_state_ref=f"passive.after.{tick}",
        effect_id=effect,
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


def test_missing_passive_baseline_has_no_causal_confidence():
    """Action reliability without a passive comparison is descriptive only."""
    assert controllability_confidence(8, 8, 0, 0) == 0.0
    assert controllability_confidence(8, 4, 0, 0) == 0.0
    assert controllability_confidence(2, 2, 0, 0) == 0.0


def test_passive_baseline_contrast():
    """Contrasting against quiet passive baseline yields full attribution up to 1.0."""
    # 8/8 hits with 8 quiet passive windows: advantage and support are 1.0.
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
    """Confidence increases with action/baseline support when contrasted."""
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
        def controllability_opportunities(
            self, effect_id, *, field, values, context_ref
        ):
            if field == "competence_id" and values == ("comp.a",):
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


def test_real_ledger_does_not_pool_another_active_source_as_passive_baseline():
    ledger = CausalEvidenceLedger(capacity=64, passive_capacity=64)
    for tick in range(8):
        _record_action(
            ledger,
            signature="signature.a",
            commitment=f"commitment.a.{tick}",
            effect="effect.shared",
            tick=tick * 2,
        )
        _record_action(
            ledger,
            signature="signature.b",
            commitment=f"commitment.b.{tick}",
            effect="effect.shared",
            tick=tick * 2 + 1,
        )

    estimate_a = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=20,
    )
    estimate_b = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.b",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=20,
    )

    assert estimate_a.action_support == estimate_b.action_support == 8
    assert estimate_a.counterfactual_support == estimate_b.counterfactual_support == 0
    assert estimate_a.counterfactual_rate is estimate_b.counterfactual_rate is None
    assert estimate_a.confidence == estimate_b.confidence == 0.0
    assert (
        sum(
            item.intervention_signature_id == "signature.b"
            for item in ledger.intervention_evidence
        )
        == 8
    )


def test_real_ledger_uses_passive_windows_not_other_active_sources_for_contrast():
    ledger = CausalEvidenceLedger(capacity=64, passive_capacity=64)
    for tick in range(8):
        _record_action(
            ledger,
            signature="signature.a",
            commitment=f"commitment.a.{tick}",
            effect="effect.shared",
            tick=tick * 3,
        )
        _record_action(
            ledger,
            signature="signature.b",
            commitment=f"commitment.b.{tick}",
            effect="effect.shared",
            tick=tick * 3 + 1,
        )
        _record_passive(ledger, effect=None, tick=tick * 3 + 2)

    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=24,
    )

    assert estimate.action_support == 8
    assert estimate.counterfactual_support == 8
    assert estimate.counterfactual_rate == 0.0
    assert estimate.confidence == 1.0


def test_real_ledger_excludes_same_commitment_multi_source_attribution():
    ledger = CausalEvidenceLedger(capacity=16, passive_capacity=16)
    for tick in range(8):
        commitment = f"commitment.coissued.{tick}"
        _record_action(
            ledger,
            signature="signature.a",
            commitment=commitment,
            effect="effect.shared",
            tick=tick * 2,
        )
        _record_action(
            ledger,
            signature="signature.b",
            commitment=commitment,
            effect="effect.shared",
            tick=tick * 2 + 1,
        )

    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=16,
    )
    assert estimate.action_support == 0
    assert estimate.confidence == 0.0


def test_agency_keeps_existing_active_alternative_semantics():
    """The #108 passive-baseline change is scoped to controllability only."""
    ledger = CausalEvidenceLedger(capacity=64, passive_capacity=64)
    for tick in range(8):
        _record_action(
            ledger,
            signature="signature.a",
            commitment=f"commitment.a.{tick}",
            effect="effect.shared",
            tick=tick * 2,
        )
        _record_action(
            ledger,
            signature="signature.b",
            commitment=f"commitment.b.{tick}",
            effect="effect.shared",
            tick=tick * 2 + 1,
        )

    estimate = AgencyModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=20,
        prediction_match=1.0,
    )

    assert estimate.support == 8
    assert estimate.counterfactual_support == 8
    assert estimate.temporal_contingency == 1.0
    assert estimate.causal_specificity == 0.0
    assert estimate.confidence == 0.0


def test_real_ledger_keeps_other_active_source_diagnostics_separate():
    ledger = CausalEvidenceLedger(capacity=16, passive_capacity=16)
    _record_action(
        ledger,
        signature="signature.a",
        commitment="commitment.a",
        effect=None,
        tick=0,
    )
    _record_action(
        ledger,
        signature="signature.b",
        commitment="commitment.b",
        effect="effect.shared",
        tick=2,
    )

    assert ledger.active_alternative_effect_opportunities(
        "effect.shared",
        field="intervention_signature_id",
        values=("signature.a",),
        context_ref="context.matched",
    ) == (1, 1)
    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=3,
    )
    assert estimate.counterfactual_support == 0
    assert estimate.confidence == 0.0


def test_real_ledger_uses_only_exact_context_for_passive_baseline():
    ledger = CausalEvidenceLedger(capacity=16, passive_capacity=16)
    _record_action(
        ledger,
        signature="signature.a",
        commitment="commitment.a",
        effect="effect.shared",
        tick=0,
    )
    ledger.observe_passive_window(
        tick_start=2,
        tick_end=3,
        context_ref="context.other",
        prior_state_ref="passive.before.other",
        resulting_state_ref="passive.after.other",
        effect_id=None,
    )

    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=3,
    )
    assert estimate.action_support == 1
    assert estimate.counterfactual_support == 0
    assert estimate.counterfactual_rate is None
    assert estimate.confidence == 0.0


def test_none_context_means_uncontextualized_rows_only():
    ledger = CausalEvidenceLedger(capacity=16, passive_capacity=16)
    _record_action(
        ledger,
        signature="signature.a",
        commitment="commitment.contextual",
        effect="effect.shared",
        tick=0,
        context="context.matched",
    )
    _record_action(
        ledger,
        signature="signature.a",
        commitment="commitment.uncontextualized",
        effect="effect.shared",
        tick=2,
    )
    _record_passive(ledger, effect=None, tick=4, context=None)
    payload = ledger.checkpoint()
    evidence_rows = payload["evidence"]
    assert isinstance(evidence_rows, list)
    evidence_rows[1]["context_ref"] = None
    ledger = CausalEvidenceLedger.restore(payload)

    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id=None,
        tick=5,
    )
    assert estimate.action_support == 1
    assert estimate.counterfactual_support == 1
    assert estimate.confidence == 0.125


def test_retained_rows_are_not_misrepresented_as_independent_commitments():
    ledger = CausalEvidenceLedger(capacity=16, passive_capacity=16)
    for tick in range(4):
        _record_action(
            ledger,
            signature="signature.a",
            commitment="commitment.same",
            effect="effect.shared",
            tick=tick * 2,
        )
        _record_passive(ledger, effect=None, tick=tick * 2 + 1)

    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=8,
    )
    # These are retained rows, not four independent commitments or trials.
    assert estimate.action_support == 4
    assert estimate.counterfactual_support == 4
    assert estimate.confidence == 0.5


def test_real_ledger_excludes_overlapping_source_windows():
    ledger = CausalEvidenceLedger(capacity=16, passive_capacity=16)
    _record_action(
        ledger,
        signature="signature.a",
        commitment="commitment.a",
        effect="effect.shared",
        tick=0,
    )
    _record_action(
        ledger,
        signature="signature.b",
        commitment="commitment.b",
        effect="effect.shared",
        tick=0,
    )

    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=1,
    )
    assert estimate.action_support == 0
    assert estimate.confidence == 0.0


def test_real_ledger_rejects_passive_window_overlapping_an_active_command():
    ledger = CausalEvidenceLedger(capacity=16, passive_capacity=16)
    _record_action(
        ledger,
        signature="signature.a",
        commitment="commitment.a",
        effect="effect.shared",
        tick=0,
    )
    _record_passive(ledger, effect=None, tick=0)

    estimate = ControllabilityModel().update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.INTERVENTION,
        source_ref="signature.a",
        effect_id="effect.shared",
        context_id="context.matched",
        tick=1,
    )
    assert estimate.action_support == 1
    assert estimate.counterfactual_support == 0
    assert estimate.counterfactual_rate is None
    assert estimate.confidence == 0.0
