from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pytest

from organism.actuation import (
    ActuatorEvidenceContext,
    ActuatorSurface,
    CompetenceDevelopmentEngine,
    OrganismExplorationComposer,
)
from organism.cognition import CognitiveRuntimeCheckpoint, CognitiveTransitionState
from organism.cognition.graph import CognitiveGraph, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.types import NodeKind
from organism.identity import OrganismIdentity
from organism.transition import (
    EmbodimentContextRef,
    OrganismLifecycle,
    OrganismTransitionState,
    WorldCausalCorrelation,
    transition_organism_lifecycle,
)


IDENTITY = OrganismIdentity.create(UUID("b52d1a22-49f1-43ad-8b27-10e355e94351"))
ACTUATORS = ("effector.left", "effector.right")


def _context(surface: ActuatorSurface) -> ActuatorEvidenceContext:
    return ActuatorEvidenceContext(
        body_id="body.1",
        embodiment_episode_id="episode.1",
        morphology_revision="morphology.r1",
        surface_fingerprint=surface.contract_fingerprint,
    )


def _transition(
    *,
    tick: int = 1,
    lifecycle: OrganismLifecycle = OrganismLifecycle.ACTIVE,
    embodiment: EmbodimentContextRef | None = None,
) -> OrganismTransitionState:
    identity = IDENTITY
    graph = CognitiveGraph(
        nodes=(PlasticNode("node.1", NodeKind.STATE),), edges=(), kernel_limits=KernelLimits()
    )
    checkpoint = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState.initial(identity), graph=graph
    ).to_organism_checkpoint()
    return OrganismTransitionState(
        owner_checkpoint=checkpoint,
        organism_step=1,
        embodiment=embodiment
        or EmbodimentContextRef("body.1", "episode.1", "morphology.r1"),
        world_correlation=WorldCausalCorrelation("world.1", tick),
        lifecycle=lifecycle,
    )


def _composer(surface: ActuatorSurface, context: ActuatorEvidenceContext) -> OrganismExplorationComposer:
    engine = CompetenceDevelopmentEngine(
        ACTUATORS,
        organism_id=str(IDENTITY),
        embodiment_fingerprint=context.fingerprint,
    )
    return OrganismExplorationComposer(organism_id=str(IDENTITY), engine=engine)


@dataclass(frozen=True)
class _CommittedCut:
    world_id: str
    causal_tick: int
    commit_id: str


def _candidate_with_command(composer: OrganismExplorationComposer, surface: ActuatorSurface, context: ActuatorEvidenceContext):
    for tick in range(1, 65):
        candidate = composer.prepare(
            _transition(tick=tick), actuator_context=context, surface=surface
        )
        if candidate.motor_command is not None:
            return candidate
    pytest.fail("deterministic bounded exploration produced no command in 64 ticks")


def test_production_composer_stages_identity_bound_exploration_without_evidence_claims() -> None:
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    composer = _composer(surface, context)

    # Candidate generation is read-only with respect to accepted owner state.
    first = _candidate_with_command(composer, surface, context)
    assert first.proposal is not None
    assert first.proposal.source.value == "exploration"
    assert first.proposal.intent_id is None
    assert first.proposal.competence_id is None
    assert first.proposal.effect_target_id is None
    assert first.proposal.evaluation.effect_confidence == 0.0
    assert first.proposal.evaluation.controllability is None
    assert first.motor_command is not None
    assert first.motor_command.surface_fingerprint == surface.contract_fingerprint
    assert {actuator_id for actuator_id, _ in first.motor_command.channels} <= set(ACTUATORS)

    # Competing candidates from the same base are possible, but only the one
    # attached to the committed World causal cut can advance owner state.
    with pytest.raises(TypeError, match="committed-cut receipt"):
        composer.adopt_after_world_commit(
            first,
            committed_cut=object(),
        )
    with pytest.raises(ValueError, match="does not match the committed"):
        composer.adopt_after_world_commit(
            first,
            committed_cut=_CommittedCut("world.1", first.causal_tick + 1, "commit.wrong"),
        )
    composer.adopt_after_world_commit(
        first,
        committed_cut=_CommittedCut("world.1", first.causal_tick, "commit.1"),
    )
    with pytest.raises(ValueError, match="stale"):
        composer.adopt_after_world_commit(
            first,
            committed_cut=_CommittedCut("world.1", first.causal_tick, "commit.1"),
        )


def test_rejects_wrong_episode_morphology_surface_and_lifecycle() -> None:
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    composer = _composer(surface, context)

    with pytest.raises(ValueError, match="does not match the accepted transition"):
        composer.prepare(
            _transition(embodiment=EmbodimentContextRef("body.1", "episode.old", "morphology.r1")),
            actuator_context=context,
            surface=surface,
        )

    wrong_surface = ActuatorSurface.from_effector_ids(ACTUATORS, fingerprint_material="changed")
    with pytest.raises(ValueError, match="fingerprint"):
        composer.prepare(
            _transition(), actuator_context=context, surface=wrong_surface
        )

    dormant = transition_organism_lifecycle(
        _transition(),
        OrganismLifecycle.DORMANT,
        transition_id="lifecycle.1",
        request_ref="test.request",
    )
    with pytest.raises(ValueError, match="ACTIVE"):
        composer.prepare(
            dormant,
            actuator_context=context,
            surface=surface,
        )


def test_surface_order_is_not_actuator_identity_authority() -> None:
    surface_a = ActuatorSurface.from_effector_ids(ACTUATORS)
    surface_b = ActuatorSurface.from_effector_ids(tuple(reversed(ACTUATORS)))
    assert surface_a.contract_fingerprint == surface_b.contract_fingerprint
    context_a = _context(surface_a)
    context_b = _context(surface_b)
    composer_a = _composer(surface_a, context_a)
    composer_b = _composer(surface_b, context_b)

    result_a = _candidate_with_command(composer_a, surface_a, context_a)
    result_b = _candidate_with_command(composer_b, surface_b, context_b)
    assert result_a.motor_command is not None
    assert result_b.motor_command is not None
    assert result_a.causal_tick == result_b.causal_tick
    assert result_a.motor_command == result_b.motor_command
