"""Unit test for somatic feedback integration in OrganismExplorationComposer.adopt_after_world_commit."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from organism import (
    EmbodimentContextRef,
    OrganismCheckpoint,
    OrganismIdentity,
    OrganismLifecycle,
    OrganismTransitionState,
    WorldCausalCorrelation,
)
from organism.actuation import (
    ActuatorEvidenceContext,
    ActuatorSurface,
    CompetenceDevelopmentEngine,
    OrganismExplorationComposer,
)
from organism.cognition import (
    CognitiveRuntimeCheckpoint,
    CognitiveTransitionState,
)
from organism.cognition.graph import CognitiveGraph
from organism.cognition.limits import KernelLimits

IDENTITY = OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000999"))
LIMITS = KernelLimits()
EMBODIMENT = EmbodimentContextRef("body-1", "episode-1", "morph-1")


@dataclass(frozen=True, slots=True)
class _CutReceipt:
    world_id: str
    causal_tick: int
    commit_id: str


def test_adopt_after_world_commit_with_somatic_feedback() -> None:
    surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material="test-fingerprint",
    )
    act_ctx = ActuatorEvidenceContext("body-1", "episode-1", "morph-1", surface.contract_fingerprint)
    engine = CompetenceDevelopmentEngine(
        surface.actuator_ids,
        organism_id=str(IDENTITY),
        embodiment_fingerprint=act_ctx.fingerprint,
    )
    composer = OrganismExplorationComposer(organism_id=str(IDENTITY), engine=engine)

    cognition = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState.initial(IDENTITY),
        graph=CognitiveGraph(nodes=(), edges=(), kernel_limits=LIMITS),
    )
    transition = OrganismTransitionState(
        owner_checkpoint=cognition.to_organism_checkpoint(),
        organism_step=0,
        embodiment=EMBODIMENT,
        world_correlation=WorldCausalCorrelation("world-1", 0),
        lifecycle=OrganismLifecycle.ACTIVE,
    )

    candidate = composer.prepare(
        transition,
        actuator_context=act_ctx,
        surface=surface,
        target_causal_tick=1,
        physical_step_id="step-1",
        commit_id="commit-1",
    )

    receipt = _CutReceipt(world_id="world-1", causal_tick=1, commit_id="commit-1")
    somatic_feedback = {"joint.hinge:pos": 0.05, "joint.hinge:vel": 0.12}

    # Verify adopt with somatic observation calls engine.observe
    composer.adopt_after_world_commit(
        candidate,
        committed_cut=receipt,
        somatic_observation=somatic_feedback,
    )

    # Engine received frame at tick 1 with exact body state
    chk = composer.engine.checkpoint()
    assert len(chk["frame_history"]) == 1
    assert chk["frame_history"][0]["tick"] == 1
    assert chk["frame_history"][0]["body_state"] == somatic_feedback
