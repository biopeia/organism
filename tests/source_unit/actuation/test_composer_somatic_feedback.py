"""Unit test for somatic feedback integration in OrganismExplorationComposer."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
import pytest

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


def _make_fixture():
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
    return surface, act_ctx, engine, composer, transition


def test_adopt_after_world_commit_with_somatic_feedback() -> None:
    surface, act_ctx, engine, composer, transition = _make_fixture()

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


def test_stage_somatic_feedback_for_candidate_atomic_durability() -> None:
    surface, act_ctx, engine, composer, transition = _make_fixture()

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

    # Invariant: live composer engine has NOT observed somatic feedback yet
    assert len(composer.engine.checkpoint()["frame_history"]) == 0

    # Stage somatic feedback into candidate state
    composer.stage_somatic_feedback_for_candidate(
        candidate,
        somatic_observation=somatic_feedback,
    )

    # Invariant: live composer engine STILL has not observed somatic feedback (no premature cognitive effects)
    assert len(composer.engine.checkpoint()["frame_history"]) == 0

    # Candidate checkpoint payload captures the staged observation
    encoded_checkpoint = composer.checkpoint_for_candidate(
        candidate,
        transition_state=transition,
    )

    # Restoring from the candidate checkpoint reproduces tick 1's somatic observation bit-identically
    restored = OrganismExplorationComposer.restore_from_committed_checkpoint(
        encoded_checkpoint,
        committed_cut=receipt,
        expected_organism_id=str(IDENTITY),
        expected_physical_step_id="step-1",
        expected_actuator_context=act_ctx,
        expected_surface=surface,
    )
    restored_chk = restored.action_composer.engine.checkpoint()
    assert len(restored_chk["frame_history"]) == 1
    assert restored_chk["frame_history"][0]["tick"] == 1
    assert restored_chk["frame_history"][0]["body_state"] == somatic_feedback

    # When World confirms commit, adopt_after_world_commit adopts staged state without duplicate observe
    composer.adopt_after_world_commit(
        candidate,
        committed_cut=receipt,
        somatic_observation=somatic_feedback,
    )
    live_chk = composer.engine.checkpoint()
    assert len(live_chk["frame_history"]) == 1
    assert live_chk["frame_history"] == restored_chk["frame_history"]


def test_stage_somatic_feedback_negative_cases() -> None:
    surface, act_ctx, engine, composer, transition = _make_fixture()

    candidate = composer.prepare(
        transition,
        actuator_context=act_ctx,
        surface=surface,
        target_causal_tick=1,
        physical_step_id="step-1",
        commit_id="commit-1",
    )

    # Rejects invalid somatic_observation type
    with pytest.raises(TypeError, match="Mapping"):
        composer.stage_somatic_feedback_for_candidate(candidate, somatic_observation="invalid")  # type: ignore

    # Stages successfully first time
    composer.stage_somatic_feedback_for_candidate(
        candidate,
        somatic_observation={"joint.hinge:pos": 0.1},
    )

    # Rejects double staging on the same candidate
    with pytest.raises(ValueError, match="already been staged"):
        composer.stage_somatic_feedback_for_candidate(
            candidate,
            somatic_observation={"joint.hinge:pos": 0.2},
        )
