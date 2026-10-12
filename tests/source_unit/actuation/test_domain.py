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
from organism.actuation.domain import ActionDomain
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
    actuations = domain.execute(command, surface)
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
        maximum_duration=None,
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
    assert domain.execute(command, surface)[0].delivered == 0.25


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
        domain.execute(forged, surface)

    domain.close(
        commitment.commitment_id,
        status=IntentStatus.FAILED,
        tick=6,
        reason="controller_terminal_failure",
    )
    with pytest.raises(RuntimeError, match="registered active commitment"):
        domain.execute(command, surface)


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

