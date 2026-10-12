from __future__ import annotations

import pytest

from organism.actuation import (
    ActuatorEvidenceContext,
    CompetenceExecutionBindingRegistry,
    CompetenceLibrary,
    CompetenceMaturity,
    derive_motor_competence_from_evidence,
)
from organism.actuation.binding import BindingStatus
from organism.actuation.competence import CompetenceGate
from organism.actuation.evidence import CausalEvidenceLedger, SensorimotorTransition
from organism.actuation.model import ControllabilityModel
from organism.actuation.sensorimotor import MotorPrimitive


def _primitive(samples: int = 8, effect_variance: float = 0.0) -> MotorPrimitive:
    sequence = ((("effector.test.1", 5),),) * 4
    return MotorPrimitive(
        primitive_id="primitive.test.1",
        embodiment_fingerprint="fingerprint.test.1",
        sequence=sequence,
        samples=samples,
        effect_mean=0.5,
        effect_variance=effect_variance,
        controllability=0.25,
        directional_consistency=0.9,
    )


def _transition(
    tick_start: int,
    tick_end: int,
    *,
    competence_id: str,
    effect_id: str | None = "effect.test.1",
    context_ref: str = "context.test.1",
) -> SensorimotorTransition:
    return SensorimotorTransition(
        transition_id=f"trans.{tick_start}.{tick_end}",
        tick_start=tick_start,
        tick_end=tick_end,
        context_ref=context_ref,
        commitment_id=f"commitment.{tick_start}",
        controller_id=f"controller.{tick_start}",
        competence_id=competence_id,
        attempt_id=f"attempt.{tick_start}",
        intervention_signature_id=f"sig.{tick_start}",
        state_before_ref=f"state.{tick_start}",
        motor_command_ref=f"command.{tick_start}",
        actuation_ref=f"actuation.{tick_start}",
        prediction_ref=None,
        state_after_ref=f"state.{tick_end}",
        observed_effect_id=effect_id,
    )


def test_derivation_from_contrasted_evidence_creates_established_competence_and_valid_binding():
    primitive = _primitive(samples=8)
    competence_id = f"competence.{primitive.primitive_id}"
    effect_id = "effect.test.1"
    applicability = ActuatorEvidenceContext(
        "body.test.1", "episode.test.1", "morphology.test.1", "surface.fp.1"
    )
    context_ref = applicability.fingerprint

    ledger = CausalEvidenceLedger()
    controllability_model = ControllabilityModel()
    library = CompetenceLibrary()
    binding_registry = CompetenceExecutionBindingRegistry()

    # 1. Record 8 action-linked interventions with effect_id
    for i in range(8):
        ledger.observe(
            _transition(
                tick_start=i * 2,
                tick_end=i * 2 + 1,
                competence_id=competence_id,
                effect_id=effect_id,
                context_ref=context_ref,
            )
        )

    # 2. Record 8 passive windows with 0 hits (clean counterfactual control)
    for i in range(8):
        ledger.observe_passive_window(
            tick_start=20 + i * 2,
            tick_end=20 + i * 2 + 1,
            context_ref=context_ref,
            prior_state_ref=f"passive.before.{i}",
            resulting_state_ref=f"passive.after.{i}",
            effect_id=None,
        )

    record = derive_motor_competence_from_evidence(
        primitive,
        effect_id=effect_id,
        ledger=ledger,
        controllability_model=controllability_model,
        context_ref=context_ref,
        applicability_context=applicability,
        tick=40,
        competence_library=library,
        binding_registry=binding_registry,
    )

    # Competence assertions
    assert record.competence.competence_id == competence_id
    assert record.competence.effect_id == effect_id
    assert record.competence.maturity in {
        CompetenceMaturity.ESTABLISHED,
        CompetenceMaturity.ROBUST,
    }
    assert library.get(competence_id) is record.competence

    # Controllability assertions
    assert record.controllability_estimate is not None
    assert record.controllability_estimate.confidence > 0.0
    assert record.controllability_estimate.reliability == 1.0

    # Binding assertions
    assert record.binding is not None
    assert record.binding.competence_id == competence_id
    assert record.binding.status is BindingStatus.VALID
    assert record.binding.surface_fingerprint == applicability.surface_fingerprint
    assert record.binding.controllability == record.controllability_estimate.confidence
    assert binding_registry.get(competence_id) is record.binding


def test_derivation_without_passive_control_yields_zero_controllability_and_no_valid_binding():
    primitive = _primitive(samples=8)
    competence_id = f"competence.{primitive.primitive_id}"
    effect_id = "effect.test.1"
    context_ref = "context.test.1"

    ledger = CausalEvidenceLedger()
    controllability_model = ControllabilityModel()
    binding_registry = CompetenceExecutionBindingRegistry()

    # Only interventions, NO passive/non-action windows recorded
    for i in range(8):
        ledger.observe(
            _transition(
                tick_start=i * 2,
                tick_end=i * 2 + 1,
                competence_id=competence_id,
                effect_id=effect_id,
                context_ref=context_ref,
            )
        )

    record = derive_motor_competence_from_evidence(
        primitive,
        effect_id=effect_id,
        ledger=ledger,
        controllability_model=controllability_model,
        context_ref=context_ref,
        tick=20,
        binding_registry=binding_registry,
    )

    # Without passive contrast, controllability MUST be 0.0 (no positive confidence without contrast)
    assert record.controllability_estimate is not None
    assert record.controllability_estimate.confidence == 0.0
    assert record.competence.evidence.controllability == 0.0

    # Under gate (min controllability > 0.002), competence maturity fails closed to CANDIDATE
    assert record.competence.maturity is CompetenceMaturity.CANDIDATE
    # And candidate maturity cannot produce a valid execution binding
    assert record.binding is None
    assert binding_registry.get(competence_id) is None


def test_derivation_with_zero_action_hits_yields_zero_controllability():
    primitive = _primitive(samples=8)
    competence_id = f"competence.{primitive.primitive_id}"
    effect_id = "effect.test.1"
    context_ref = "context.test.1"

    ledger = CausalEvidenceLedger()
    controllability_model = ControllabilityModel()

    # Interventions occurred, but none produced effect_id
    for i in range(8):
        ledger.observe(
            _transition(
                tick_start=i * 2,
                tick_end=i * 2 + 1,
                competence_id=competence_id,
                effect_id="effect.other",
                context_ref=context_ref,
            )
        )

    record = derive_motor_competence_from_evidence(
        primitive,
        effect_id=effect_id,
        ledger=ledger,
        controllability_model=controllability_model,
        context_ref=context_ref,
        tick=20,
    )

    assert record.controllability_estimate is not None
    assert record.controllability_estimate.confidence == 0.0
    assert record.competence.evidence.support == 0
    assert record.competence.maturity is CompetenceMaturity.CANDIDATE
    assert record.binding is None


def test_binding_requires_explicit_current_applicability_context():
    primitive = _primitive(samples=8)
    competence_id = f"competence.{primitive.primitive_id}"
    context = ActuatorEvidenceContext(
        "body.test.1", "episode.test.1", "morphology.test.1", "surface.fp.1"
    )
    ledger = CausalEvidenceLedger()
    for i in range(8):
        ledger.observe(
            _transition(
                i * 2,
                i * 2 + 1,
                competence_id=competence_id,
                context_ref=context.fingerprint,
            )
        )
    for i in range(8):
        ledger.observe_passive_window(
            tick_start=20 + i * 2,
            tick_end=21 + i * 2,
            context_ref=context.fingerprint,
            prior_state_ref=f"passive.before.{i}",
            resulting_state_ref=f"passive.after.{i}",
            effect_id=None,
        )

    registry = CompetenceExecutionBindingRegistry()
    record = derive_motor_competence_from_evidence(
        primitive,
        effect_id="effect.test.1",
        ledger=ledger,
        controllability_model=ControllabilityModel(),
        context_ref=context.fingerprint,
        tick=40,
        binding_registry=registry,
    )

    assert record.competence.maturity in {
        CompetenceMaturity.ESTABLISHED,
        CompetenceMaturity.ROBUST,
    }
    assert record.binding is None
    assert registry.get(competence_id) is None


def test_evidence_refs_and_binding_are_scoped_to_current_context():
    primitive = _primitive(samples=8)
    competence_id = f"competence.{primitive.primitive_id}"
    current = ActuatorEvidenceContext(
        "body.current", "episode.current", "morphology.current", "surface.current"
    )
    prior = ActuatorEvidenceContext(
        "body.prior", "episode.prior", "morphology.prior", "surface.prior"
    )
    ledger = CausalEvidenceLedger()
    for i in range(8):
        ledger.observe(
            _transition(
                i * 2,
                i * 2 + 1,
                competence_id=competence_id,
                context_ref=prior.fingerprint,
            )
        )
    for i in range(8):
        ledger.observe(
            _transition(
                100 + i * 2,
                101 + i * 2,
                competence_id=competence_id,
                context_ref=current.fingerprint,
            )
        )
    for i in range(8):
        ledger.observe_passive_window(
            tick_start=200 + i * 2,
            tick_end=201 + i * 2,
            context_ref=current.fingerprint,
            prior_state_ref=f"passive.current.before.{i}",
            resulting_state_ref=f"passive.current.after.{i}",
            effect_id=None,
        )

    registry = CompetenceExecutionBindingRegistry()
    record = derive_motor_competence_from_evidence(
        primitive,
        effect_id="effect.test.1",
        ledger=ledger,
        controllability_model=ControllabilityModel(),
        context_ref=current.fingerprint,
        applicability_context=current,
        tick=300,
        binding_registry=registry,
    )

    current_ids = {
        item.evidence_id
        for item in ledger.intervention_evidence
        if item.context_ref == current.fingerprint
    }
    prior_ids = {
        item.evidence_id
        for item in ledger.intervention_evidence
        if item.context_ref == prior.fingerprint
    }
    assert record.competence.evidence.controllability_evidence_refs
    assert set(record.competence.evidence.controllability_evidence_refs) <= current_ids
    assert not (set(record.competence.evidence.controllability_evidence_refs) & prior_ids)
    assert record.binding is not None
    assert set(record.binding.evidence_refs) <= current_ids
    assert record.binding.surface_fingerprint == current.surface_fingerprint
    assert record.binding.applicability_fingerprint == current.fingerprint


def test_mismatched_context_ref_and_applicability_are_rejected():
    context = ActuatorEvidenceContext(
        "body.test.1", "episode.test.1", "morphology.test.1", "surface.fp.1"
    )
    with pytest.raises(ValueError, match="context_ref must match"):
        derive_motor_competence_from_evidence(
            _primitive(),
            effect_id="effect.test.1",
            ledger=CausalEvidenceLedger(),
            controllability_model=ControllabilityModel(),
            context_ref="unrelated.context",
            applicability_context=context,
            tick=0,
        )
