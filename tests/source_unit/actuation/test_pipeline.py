from __future__ import annotations

from uuid import UUID

import pytest

from organism.actuation import (
    ActionDomain,
    AdmittedActionPlan,
    ExecutedActionPipelineResult,
    SequenceController,
    compose_action_from_admitted_decision,
    derive_actuator_constitution,
)
from organism.actuation.action import ActionSource
from organism.actuation.types import MotorIntent
from organism.agency.affordance import ActionAffordance
from organism.agency.intention import IntentStatus
from organism.agency.prospective import ProspectiveDecision
from organism.checkpoint import OrganismCheckpoint
from organism.cognition.graph import (
    CognitiveGraph,
    PlasticEdge,
    PlasticNode,
)
from organism.cognition import (
    CognitiveRuntimeCheckpoint,
    CognitiveTransitionState,
)
from organism.cognition.limits import KernelLimits
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity
from organism.transition import (
    EmbodimentContextRef,
    OrganismLifecycle,
    OrganismTransitionState,
    WorldCausalCorrelation,
    transition_organism_lifecycle,
)


IDENTITY = OrganismIdentity.create(UUID("11111111-1111-1111-1111-111111111111"))
EMBODIMENT = EmbodimentContextRef(
    body_id="body.1",
    embodiment_episode_id="episode.1",
    morphology_revision="morph.r1",
)
LIMITS = KernelLimits()


def _transition_state(
    step: int = 5,
    tick: int = 10,
    lifecycle: OrganismLifecycle = OrganismLifecycle.ACTIVE,
    embodiment: EmbodimentContextRef | None = EMBODIMENT,
    world_correlation: WorldCausalCorrelation | None = None,
) -> OrganismTransitionState:
    graph = CognitiveGraph(
        nodes=(PlasticNode("node.1", NodeKind.STATE),),
        edges=(),
        kernel_limits=LIMITS,
    )
    cognition_state = CognitiveTransitionState.initial(IDENTITY)
    runtime = CognitiveRuntimeCheckpoint(state=cognition_state, graph=graph)
    owner_checkpoint = runtime.to_organism_checkpoint()
    return OrganismTransitionState(
        owner_checkpoint=owner_checkpoint,
        organism_step=step,
        embodiment=embodiment,
        world_correlation=(
            world_correlation
            if world_correlation is not None
            else WorldCausalCorrelation("world.test", tick)
        ),
        lifecycle=lifecycle,
    )


def _decision() -> ProspectiveDecision:
    return ProspectiveDecision(
        competence_id="competence.reach",
        anticipated_effect_id="effect.displacement",
        prediction_ref="prediction.reach.1",
        confidence=0.8,
        epistemic_relevance=0.7,
        homeostatic_relevance=0.3,
        origin_refs=("readout.node.1",),
        supporting_affordance_id="affordance.reach.1",
    )


def _affordance(
    embodiment_id: str | None = "episode.1",
    context_ref: str = "context.reach.1",
) -> ActionAffordance:
    return ActionAffordance(
        affordance_id="affordance.reach.1",
        competence_id="competence.reach",
        anticipated_effect_id="effect.displacement",
        context_ref=context_ref,
        embodiment_id=embodiment_id,
        prediction_confidence=0.8,
        controllability=0.6,
        executability_confidence=1.0,
        prediction_ref="prediction.reach.1",
        evidence_refs=("evidence.reach.1",),
    )


def _plan(
    decision: ProspectiveDecision | None = None,
    affordance: ActionAffordance | None = None,
    **overrides: object,
) -> AdmittedActionPlan:
    kwargs: dict[str, object] = {
        "decision": decision or _decision(),
        "supporting_affordance": affordance or _affordance(),
        "intent_id": "intent.action-30.1",
        "proposal_id": "proposal.action-30.1",
        "commitment_id": "commitment.action-30.1",
        "controller_id": "controller.action-30.1",
        "command_id": "command.action-30.1",
        "intervention_signature_id": "signature.action-30.1",
        "interruptibility": 1.0,
        "minimum_duration": 0,
        "maximum_duration": None,
    }
    kwargs.update(overrides)
    return AdmittedActionPlan(**kwargs)  # type: ignore[arg-type]


def test_compose_action_connects_accepted_transition_to_motor_command_and_attempt() -> None:
    trans_state = _transition_state(step=5, tick=12)
    domain = ActionDomain()
    surface = derive_actuator_constitution(("effector.joint.1",))
    controller = SequenceController(
        controller_id="controller.action-30.1",
        sequence=((MotorIntent("effector.joint.1", 0.65),),),
    )
    plan = _plan()

    result = compose_action_from_admitted_decision(
        trans_state,
        domain,
        plan,
        controller=controller,
        surface=surface,
    )

    # 1. Intent is registered and ACTIVE
    assert result.intent.intent_id == "intent.action-30.1"
    assert result.intent.status is IntentStatus.ACTIVE
    assert result.intent.competence_id == "competence.reach"
    assert result.intent.created_tick == 12

    # 2. Proposal matches decision evaluation
    assert result.proposal.proposal_id == "proposal.action-30.1"
    assert result.proposal.source is ActionSource.PROSPECTION
    assert result.proposal.competence_id == "competence.reach"
    assert result.proposal.effect_target_id == "effect.displacement"
    assert result.proposal.evaluation.effect_confidence == pytest.approx(0.8)
    assert result.proposal.evaluation.controllability == pytest.approx(0.6)
    assert result.proposal.intent_id == "intent.action-30.1"

    # 3. Commitment is active in the domain
    assert result.commitment.commitment_id == "commitment.action-30.1"
    assert result.commitment.active
    assert result.commitment.started_tick == 12
    assert domain.active_commitment is result.commitment

    # 4. MotorCommand bears full provenance
    assert result.motor_command.command_id == "command.action-30.1"
    assert result.motor_command.commitment_id == "commitment.action-30.1"
    assert result.motor_command.controller_id == "controller.action-30.1"
    assert result.motor_command.competence_id == "competence.reach"
    assert result.motor_command.surface_fingerprint == surface.contract_fingerprint
    assert result.motor_command.issued_at_tick == 12
    assert result.motor_command.channels == (("effector.joint.1", 0.65),)

    # 5. ActionAttempt records execution cross-boundary reference
    assert result.attempt.attempt_id.startswith("attempt.")
    assert result.attempt.commitment_id == "commitment.action-30.1"
    assert result.attempt.motor_command_ref == "command.action-30.1"
    assert result.attempt.started_tick == 12
    assert result.attempt.open


def test_compose_action_fails_closed_when_lifecycle_not_active() -> None:
    active_state = _transition_state(step=5, tick=12)
    dormant_state = transition_organism_lifecycle(
        active_state,
        OrganismLifecycle.DORMANT,
        transition_id="trans-dormant-1",
        request_ref="presence-terminated",
    )
    domain = ActionDomain()
    surface = derive_actuator_constitution(("effector.joint.1",))
    controller = SequenceController(
        controller_id="controller.action-30.1",
        sequence=((MotorIntent("effector.joint.1", 0.5),),),
    )

    with pytest.raises(ValueError, match="requires an ACTIVE OrganismTransitionState"):
        compose_action_from_admitted_decision(
            dormant_state,
            domain,
            _plan(),
            controller=controller,
            surface=surface,
        )

    assert domain.active_commitment is None


def test_compose_action_fails_closed_when_embodiment_is_missing_or_mismatched() -> None:
    no_body_state = _transition_state(step=5, tick=12, embodiment=None)
    domain = ActionDomain()
    surface = derive_actuator_constitution(("effector.joint.1",))
    controller = SequenceController(
        controller_id="controller.action-30.1",
        sequence=((MotorIntent("effector.joint.1", 0.5),),),
    )

    with pytest.raises(ValueError, match="explicit Embodiment context"):
        compose_action_from_admitted_decision(
            no_body_state,
            domain,
            _plan(),
            controller=controller,
            surface=surface,
        )

    # Mismatched embodiment between affordance and transition state
    mismatched_affordance = _affordance(embodiment_id="episode.DIFFERENT")
    active_state = _transition_state(step=5, tick=12)

    with pytest.raises(ValueError, match="embodiment_id does not match"):
        compose_action_from_admitted_decision(
            active_state,
            domain,
            _plan(affordance=mismatched_affordance),
            controller=controller,
            surface=surface,
        )

    assert domain.active_commitment is None


def test_compose_action_fails_closed_when_decision_contradicts_affordance() -> None:
    active_state = _transition_state(step=5, tick=12)
    domain = ActionDomain()
    surface = derive_actuator_constitution(("effector.joint.1",))
    controller = SequenceController(
        controller_id="controller.action-30.1",
        sequence=((MotorIntent("effector.joint.1", 0.5),),),
    )

    # Decision points to a different affordance ID
    corrupt_decision = ProspectiveDecision(
        competence_id="competence.reach",
        anticipated_effect_id="effect.displacement",
        prediction_ref="prediction.reach.1",
        confidence=0.8,
        epistemic_relevance=0.7,
        homeostatic_relevance=0.3,
        origin_refs=("readout.node.1",),
        supporting_affordance_id="affordance.FOREIGN",
    )

    with pytest.raises(ValueError, match="does not reference the supplied affordance"):
        compose_action_from_admitted_decision(
            active_state,
            domain,
            _plan(decision=corrupt_decision),
            controller=controller,
            surface=surface,
        )

    assert domain.active_commitment is None
