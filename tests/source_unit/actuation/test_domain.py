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

