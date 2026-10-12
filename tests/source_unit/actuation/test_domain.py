from __future__ import annotations

import pytest

from organism.agency.executive_outcome import (
    ExecutiveOutcomeLedger,
    ExecutiveOutcomeSample,
    OutcomeClass,
)
from organism.agency.affordance import ActionAffordance
from organism.agency.intention import IntentStatus
from organism.agency.prospective import ProspectiveDecision
from organism.actuation.action import (
    ActionEvaluation,
    ActionJustification,
    ActionProposal,
    ActionSource,
)
from organism.actuation.commitment import ActionCommitment
from organism.actuation.domain import ActionDomain
from organism.actuation.commitment import CommitmentStatus
from organism.actuation.surface import derive_actuator_constitution
from organism.actuation.types import MotorIntent
from organism.actuation.controller import SequenceController


def _decision() -> ProspectiveDecision:
    return ProspectiveDecision(
        competence_id="competence.walk",
        anticipated_effect_id="effect.displacement",
        prediction_ref="prediction.1",
        confidence=0.5,
        epistemic_relevance=0.4,
        homeostatic_relevance=0.2,
        origin_refs=("source.readout",),
        supporting_affordance_id="affordance.1",
    )


def _affordance() -> ActionAffordance:
    return ActionAffordance(
        affordance_id="affordance.1",
        competence_id="competence.walk",
        anticipated_effect_id="effect.displacement",
        context_ref="context.1",
        embodiment_id="embodiment.1",
        prediction_confidence=0.5,
        controllability=0.4,
        executability_confidence=1.0,
        prediction_ref="prediction.1",
        evidence_refs=("evidence.1",),
    )


def _proposal(intent_id: str = "intent.1") -> ActionProposal:
    return ActionProposal(
        proposal_id="proposal.1",
        source=ActionSource.COMPETENCE,
        effect_target_id="effect.displacement",
        competence_id="competence.walk",
        justification=ActionJustification(
            effect_target_id="effect.displacement",
            competence_id="competence.walk",
        ),
        evaluation=ActionEvaluation(effect_confidence=0.5, uncertainty=0.2),
        intent_id=intent_id,
    )


def _active_domain():
    surface = derive_actuator_constitution(("effector.left",))
    domain = ActionDomain(outcome_ledger=ExecutiveOutcomeLedger())
    intent = domain.admit(
        _decision(),
        supporting_affordance=_affordance(),
        intent_id="intent.1",
        context_ref="context.1",
        embodiment_id="embodiment.1",
        tick=4,
    )
    selected = domain.select_and_commit(
        (_proposal(),),
        tick=4,
        commitment_id="commitment.1",
        controller_id="controller.1",
        surface=surface,
        embodiment_id="embodiment.1",
        interruptibility=1.0,
        minimum_duration=0,
        maximum_duration=None,
    )
    return domain, intent, selected.commitment, surface


def test_admitted_action_composes_to_registered_actuation_and_explicit_close():
    domain, intent, commitment, surface = _active_domain()
    assert intent.status is IntentStatus.ACTIVE
    assert commitment is domain.active_commitment

    controller = SequenceController(
        controller_id="controller.1",
        sequence=((MotorIntent("effector.left", 0.75),),),
    )
    command = domain.issue_command(controller, tick=5, command_id="command.1")
    actuations = domain.execute(command, surface, tick=5)
    assert [(item.actuator_id, item.delivered) for item in actuations] == [
        ("effector.left", 0.75)
    ]

    sample = ExecutiveOutcomeSample(
        tick=6,
        outcome_class=OutcomeClass.POSITIVE,
        reason="anticipated_effect_observed",
        effect_similarity=1.0,
        prediction_mismatch=0.0,
        progress_before_failure=None,
    )
    domain.close(
        commitment.commitment_id,
        status=IntentStatus.SATISFIED,
        tick=6,
        reason="anticipated_effect_observed",
        outcome=sample,
    )
    assert intent.status is IntentStatus.SATISFIED
    assert not commitment.active
    assert domain.active_commitment is None
    assert domain._outcome_ledger.get(("competence.walk", "effect.displacement")).satisfactions == 1

    with pytest.raises(ValueError, match="already been closed"):
        domain.close(
            commitment.commitment_id,
            status=IntentStatus.SATISFIED,
            tick=7,
            reason="duplicate",
        )


def test_exploration_proposal_without_intent_can_commit_and_issue_command():
    surface = derive_actuator_constitution(("effector.left",))
    domain = ActionDomain()
    proposal = ActionProposal(
        proposal_id="exploration.1",
        source=ActionSource.EXPLORATION,
        effect_target_id=None,
        competence_id=None,
        justification=ActionJustification(),
        evaluation=ActionEvaluation(),
    )

    selected = domain.select_and_commit(
        (proposal,),
        tick=7,
        commitment_id="commitment.exploration.1",
        controller_id="controller.exploration.1",
        surface=surface,
        embodiment_id="embodiment.1",
        interruptibility=1.0,
        minimum_duration=0,
        maximum_duration=2,
    )
    commitment = selected.commitment
    assert commitment is not None
    assert commitment.intent_id is None
    assert commitment.competence_id is None
    assert commitment.effect_target_id is None

    controller = SequenceController(
        controller_id="controller.exploration.1",
        sequence=((MotorIntent("effector.left", 0.25),),),
    )
    command = domain.issue_command(controller, tick=7, command_id="command.exploration.1")
    assert command.competence_id is None
    assert command.channels == (("effector.left", 0.25),)
    assert domain.execute(command, surface, tick=7)[0].delivered == 0.25
    assert domain.issue_command(controller, tick=8, command_id="command.exploration.2")
    with pytest.raises(RuntimeError, match="has expired"):
        domain.issue_command(controller, tick=9, command_id="command.exploration.3")
    with pytest.raises(RuntimeError, match="has expired"):
        domain.execute(command, surface, tick=9)
    domain.close_exploration(
        commitment.commitment_id,
        tick=9,
        reason="exploration_window_elapsed",
    )
    assert not commitment.active
    assert commitment.status is CommitmentStatus.INVALIDATED


@pytest.mark.parametrize(
    "maximum_duration",
    (None, 0, 1.5, True),
)
def test_intent_free_exploration_requires_positive_duration(maximum_duration):
    surface = derive_actuator_constitution(("effector.left",))
    domain = ActionDomain()
    proposal = ActionProposal(
        proposal_id="exploration.unbounded",
        source=ActionSource.EXPLORATION,
        effect_target_id=None,
        competence_id=None,
        justification=ActionJustification(),
        evaluation=ActionEvaluation(),
    )

    with pytest.raises(ValueError, match="positive maximum duration"):
        domain.select_and_commit(
            (proposal,),
            tick=7,
            commitment_id="commitment.exploration.unbounded",
            controller_id="controller.exploration.unbounded",
            surface=surface,
            embodiment_id="embodiment.1",
            interruptibility=1.0,
            minimum_duration=0,
            maximum_duration=maximum_duration,
        )


@pytest.mark.parametrize(
    "justification,evaluation",
    (
        (ActionJustification(prediction_id="prediction.1"), ActionEvaluation()),
        (ActionJustification(evidence_refs=("evidence.1",)), ActionEvaluation()),
        (ActionJustification(), ActionEvaluation(effect_confidence=0.1)),
        (ActionJustification(), ActionEvaluation(controllability=0.1)),
    ),
)
def test_intent_free_exploration_rejects_causal_evidence_claims(justification, evaluation):
    surface = derive_actuator_constitution(("effector.left",))
    domain = ActionDomain()
    proposal = ActionProposal(
        proposal_id="exploration.unsupported-evidence",
        source=ActionSource.EXPLORATION,
        effect_target_id=None,
        competence_id=None,
        justification=justification,
        evaluation=evaluation,
    )

    with pytest.raises(ValueError, match="cannot claim causal evidence"):
        domain.select_and_commit(
            (proposal,),
            tick=7,
            commitment_id="commitment.exploration.unsupported-evidence",
            controller_id="controller.exploration.unsupported-evidence",
            surface=surface,
            embodiment_id="embodiment.1",
            interruptibility=1.0,
            minimum_duration=0,
            maximum_duration=2,
        )


@pytest.mark.parametrize(
    "source",
    (ActionSource.COMPETENCE, ActionSource.PROSPECTION, ActionSource.PROTECTION),
)
def test_non_exploration_proposal_without_intent_is_rejected(source):
    surface = derive_actuator_constitution(("effector.left",))
    domain = ActionDomain()
    proposal = ActionProposal(
        proposal_id="proposal.without.intent",
        source=source,
        effect_target_id=None,
        competence_id=None,
        justification=ActionJustification(),
        evaluation=ActionEvaluation(),
    )

    with pytest.raises(ValueError, match="only exploration"):
        domain.select_and_commit(
            (proposal,),
            tick=7,
            commitment_id="commitment.invalid",
            controller_id="controller.invalid",
            surface=surface,
            embodiment_id="embodiment.1",
            interruptibility=1.0,
            minimum_duration=0,
            maximum_duration=None,
    )


def test_action_domain_checkpoint_restores_active_intent_commitment_and_outcomes():
    domain, intent, commitment, _ = _active_domain()
    checkpoint = domain.checkpoint()

    restored = ActionDomain.restore(checkpoint)

    assert restored.checkpoint() == checkpoint
    assert restored.active_commitment is not None
    assert restored.active_commitment.commitment_id == commitment.commitment_id
    restored_intent = restored._intents[intent.intent_id]
    assert restored_intent.status is IntentStatus.ACTIVE


def test_restored_active_action_can_continue_and_record_owned_outcome():
    domain, _, commitment, _ = _active_domain()
    restored = ActionDomain.restore(domain.checkpoint())
    sample = ExecutiveOutcomeSample(
        tick=6,
        outcome_class=OutcomeClass.POSITIVE,
        reason="anticipated_effect_observed",
        effect_similarity=1.0,
        prediction_mismatch=0.0,
        progress_before_failure=None,
    )

    restored.close(
        commitment.commitment_id,
        status=IntentStatus.SATISFIED,
        tick=6,
        reason="anticipated_effect_observed",
        outcome=sample,
    )

    assert restored.active_commitment is None
    assert restored._intents["intent.1"].status is IntentStatus.SATISFIED
    assert (
        restored._outcome_ledger.get(("competence.walk", "effect.displacement")).satisfactions
        == 1
    )
    # Restoring creates an independent continuation; it does not mutate the source object.
    assert domain.active_commitment is not None


def test_action_domain_checkpoint_restores_terminal_lifecycle_and_closed_ids():
    domain, intent, commitment, _ = _active_domain()
    domain.close(
        commitment.commitment_id,
        status=IntentStatus.INVALIDATED,
        tick=5,
        reason="external_context_invalidated",
    )

    checkpoint = domain.checkpoint()
    restored = ActionDomain.restore(checkpoint)

    assert restored.checkpoint() == checkpoint
    assert restored.active_commitment is None
    assert restored._intents[intent.intent_id].status is IntentStatus.INVALIDATED
    assert commitment.commitment_id in restored._closed_commitments


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(active_commitment_id="missing"),
        lambda payload: payload["commitments"].clear(),
        lambda payload: payload["closed_commitment_ids"].append("missing"),
        lambda payload: payload["commitments"][0].update(intent_id="missing"),
    ),
)
def test_action_domain_restore_rejects_incomplete_or_inconsistent_state(mutate):
    domain, *_ = _active_domain()
    checkpoint = domain.checkpoint()
    mutate(checkpoint)

    with pytest.raises((TypeError, ValueError)):
        ActionDomain.restore(checkpoint)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["commitments"][0].update(started_tick=99),
        lambda payload: payload["commitments"][0].update(intent_id=None),
    ),
)
def test_action_domain_restore_rejects_missing_or_mismatched_intent_provenance(mutate):
    domain, *_ = _active_domain()
    checkpoint = domain.checkpoint()
    mutate(checkpoint)

    with pytest.raises(ValueError):
        ActionDomain.restore(checkpoint)


def test_action_domain_restore_preserves_explicit_bounded_exploration_commitment():
    domain = ActionDomain()
    checkpoint = domain.checkpoint()
    commitment = ActionCommitment(
        commitment_id="commitment.exploration",
        proposal_id="proposal.exploration",
        effect_target_id=None,
        competence_id=None,
        started_tick=2,
        controller_id="controller.exploration",
        surface_fingerprint="surface.sha256:exploration",
        proposal_source=ActionSource.EXPLORATION,
        maximum_duration=1,
    )
    checkpoint["commitments"] = [commitment.checkpoint()]
    checkpoint["active_commitment_id"] = commitment.commitment_id

    restored = ActionDomain.restore(checkpoint)

    assert restored.active_commitment is not None
    assert restored.active_commitment.proposal_source is ActionSource.EXPLORATION
    assert restored.active_commitment.intent_id is None
    assert restored.active_commitment.maximum_duration == 1


@pytest.mark.parametrize(
    "mutate", ("termination_reason", "duplicate_reference", "ended_before_progress")
)
def test_action_domain_restore_rejects_conflicting_terminal_intent_joins(mutate):
    domain, intent, commitment, _ = _active_domain()
    domain.close(
        commitment.commitment_id,
        status=IntentStatus.FAILED,
        tick=6,
        reason="controller_failed",
    )
    checkpoint = domain.checkpoint()
    if mutate == "termination_reason":
        checkpoint["intents"][0]["termination_reason"] = "different_reason"
    elif mutate == "duplicate_reference":
        duplicate = dict(checkpoint["commitments"][0])
        duplicate["commitment_id"] = "commitment.duplicate"
        checkpoint["commitments"].append(duplicate)
        checkpoint["closed_commitment_ids"].append("commitment.duplicate")
    else:
        checkpoint["commitments"][0]["ended_tick"] = intent.last_progress_tick - 1

    with pytest.raises(ValueError):
        ActionDomain.restore(checkpoint)


def test_domain_rejects_fabricated_intent_and_mismatched_proposal():
    domain = ActionDomain()
    surface = derive_actuator_constitution(("effector.left",))
    with pytest.raises(ValueError, match="registered intent"):
        domain.select_and_commit(
            (_proposal("intent.fabricated"),),
            tick=4,
            commitment_id="commitment.1",
            controller_id="controller.1",
            surface=surface,
            embodiment_id="embodiment.1",
            interruptibility=1.0,
            minimum_duration=0,
            maximum_duration=None,
        )

    domain.admit(
        _decision(),
        supporting_affordance=_affordance(),
        intent_id="intent.1",
        context_ref="context.1",
        embodiment_id="embodiment.1",
        tick=4,
    )
    mismatched = ActionProposal(
        proposal_id="proposal.bad",
        source=ActionSource.COMPETENCE,
        effect_target_id="effect.other",
        competence_id="competence.walk",
        justification=ActionJustification(),
        evaluation=ActionEvaluation(),
        intent_id="intent.1",
    )
    with pytest.raises(ValueError, match="effect does not match"):
        domain.select_and_commit(
            (mismatched,),
            tick=4,
            commitment_id="commitment.1",
            controller_id="controller.1",
            surface=surface,
            embodiment_id="embodiment.1",
            interruptibility=1.0,
            minimum_duration=0,
            maximum_duration=None,
        )


def test_admission_rejects_decision_confidence_not_supported_by_affordance():
    domain = ActionDomain()
    decision = _decision()
    unsupported = ProspectiveDecision(
        competence_id=decision.competence_id,
        anticipated_effect_id=decision.anticipated_effect_id,
        prediction_ref=decision.prediction_ref,
        confidence=0.9,
        epistemic_relevance=decision.epistemic_relevance,
        homeostatic_relevance=decision.homeostatic_relevance,
        origin_refs=decision.origin_refs,
        supporting_affordance_id=decision.supporting_affordance_id,
    )

    with pytest.raises(ValueError, match="does not match its supporting affordance"):
        domain.admit(
            unsupported,
            supporting_affordance=_affordance(),
            intent_id="intent.1",
            context_ref="context.1",
            embodiment_id="embodiment.1",
            tick=4,
        )


def test_domain_rejects_unknown_stale_or_provenance_mismatched_command():
    domain, _, commitment, surface = _active_domain()
    controller = SequenceController("controller.1", ((MotorIntent("effector.left", 0.5),),))
    command = domain.issue_command(controller, tick=5, command_id="command.1")
    forged = type(command)(
        command_id="command.forged",
        commitment_id="commitment.unknown",
        controller_id=command.controller_id,
        competence_id=command.competence_id,
        surface_fingerprint=command.surface_fingerprint,
        channels=command.channels,
        issued_at_tick=command.issued_at_tick,
        embodiment_id=command.embodiment_id,
    )
    with pytest.raises(RuntimeError, match="registered active commitment"):
        domain.execute(forged, surface, tick=5)

    domain.close(
        commitment.commitment_id,
        status=IntentStatus.FAILED,
        tick=6,
        reason="controller_terminal_failure",
    )
    with pytest.raises(RuntimeError, match="registered active commitment"):
        domain.execute(command, surface, tick=6)


def test_no_proposal_does_not_issue_or_create_a_commitment():
    domain = ActionDomain()
    surface = derive_actuator_constitution(("effector.left",))
    selection = domain.select_and_commit(
        (),
        tick=1,
        commitment_id="commitment.1",
        controller_id="controller.1",
        surface=surface,
        embodiment_id="embodiment.1",
        interruptibility=1.0,
        minimum_duration=0,
        maximum_duration=None,
    )
    assert selection.commitment is None
    assert domain.active_commitment is None


@pytest.mark.parametrize("tick", [True, 4.0, -1])
def test_admission_rejects_non_integer_or_negative_causal_tick(tick):
    domain = ActionDomain()
    with pytest.raises(ValueError, match="integer causal tick"):
        domain.admit(
            _decision(),
            supporting_affordance=_affordance(),
            intent_id="intent.1",
            context_ref="context.1",
            embodiment_id="embodiment.1",
            tick=tick,
        )


@pytest.mark.parametrize("tick", [True, 4.0, 3])
def test_selection_rejects_invalid_tick_or_tick_before_intent_creation(tick):
    domain = ActionDomain()
    surface = derive_actuator_constitution(("effector.left",))
    domain.admit(
        _decision(),
        supporting_affordance=_affordance(),
        intent_id="intent.1",
        context_ref="context.1",
        embodiment_id="embodiment.1",
        tick=4,
    )
    with pytest.raises(ValueError, match="integer causal tick"):
        domain.select_and_commit(
            (_proposal(),),
            tick=tick,
            commitment_id="commitment.1",
            controller_id="controller.1",
            surface=surface,
            embodiment_id="embodiment.1",
            interruptibility=1.0,
            minimum_duration=0,
            maximum_duration=None,
        )


def test_close_rejects_outcome_that_contradicts_terminal_status_before_mutation():
    domain, intent, commitment, _ = _active_domain()
    contradictory = ExecutiveOutcomeSample(
        tick=6,
        outcome_class=OutcomeClass.POSITIVE,
        reason="repeated_high_mismatch",
        effect_similarity=1.0,
        prediction_mismatch=0.0,
        progress_before_failure=None,
    )
    with pytest.raises(ValueError, match="contradicts terminal status/reason"):
        domain.close(
            commitment.commitment_id,
            status=IntentStatus.FAILED,
            tick=6,
            reason="repeated_high_mismatch",
            outcome=contradictory,
        )
    assert intent.status is IntentStatus.ACTIVE
    assert commitment.active
    assert domain.active_commitment is commitment
    assert domain._outcome_ledger.get(("competence.walk", "effect.displacement")) is None


def test_close_rejects_mismatched_reason_and_non_integer_tick_without_mutation():
    domain, intent, commitment, _ = _active_domain()
    sample = ExecutiveOutcomeSample(
        tick=6,
        outcome_class=OutcomeClass.POSITIVE,
        reason="anticipated_effect_observed",
        effect_similarity=1.0,
        prediction_mismatch=0.0,
        progress_before_failure=None,
    )
    with pytest.raises(ValueError, match="outcome reason"):
        domain.close(
            commitment.commitment_id,
            status=IntentStatus.SATISFIED,
            tick=6,
            reason="different_reason",
            outcome=sample,
        )
    with pytest.raises(ValueError, match="integer causal tick"):
        domain.close(
            commitment.commitment_id,
            status=IntentStatus.SATISFIED,
            tick=True,
            reason="anticipated_effect_observed",
        )
    assert intent.status is IntentStatus.ACTIVE
    assert commitment.active


def test_interrupted_commitment_can_be_closed_without_outcome_evidence():
    domain, intent, commitment, _ = _active_domain()
    domain.close(
        commitment.commitment_id,
        status=IntentStatus.INTERRUPTED,
        tick=6,
        reason="operator_interruption",
    )
    assert intent.status is IntentStatus.INTERRUPTED
    assert commitment.status.value == "interrupted"
    assert domain.active_commitment is None

