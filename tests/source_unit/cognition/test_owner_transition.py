from __future__ import annotations

from dataclasses import replace
from uuid import UUID

import pytest

from organism import (
    CheckpointDependency,
    EmbodimentContextRef,
    OrganismCheckpoint,
    OrganismIdentity,
    OrganismLifecycle,
    OrganismTransitionError,
    OrganismTransitionState,
    WorldCausalCorrelation,
)
from organism.cognition import (
    CognitiveRuntimeCheckpoint,
    CognitiveTransitionState,
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    advance_organism_from_sense_signals,
)
from organism.cognition.portable_checkpoint import CognitiveRuntimeCheckpointError
from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.sense_input import (
    SCALAR_FIELD_CONTRACT_ID,
    SCALAR_FIELD_CONTRACT_REVISION,
    SCALAR_FIELD_QUANTITY_KIND,
    SCALAR_FIELD_UNIT,
)
from organism.cognition.types import EdgeKind, NodeKind


IDENTITY = OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000123"))
LIMITS = KernelLimits()
EMBODIMENT = EmbodimentContextRef("body-1", "episode-2", "morph-7")


def _graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=(
            PlasticNode("sense.channel", NodeKind.SENSE),
            PlasticNode("private.state", NodeKind.STATE),
        ),
        edges=(
            PlasticEdge(
                "sense.channel", "private.state", EdgeKind.EXCITATORY,
                1.0, 0.0, 0,
            ),
        ),
        kernel_limits=LIMITS,
    )


def _provenance(**overrides: object) -> SenseProvenance:
    values: dict[str, object] = {
        "world_id": "world-1",
        "causal_tick": 10,
        "physical_step_id": "physical-step-10",
        "environment_id": "environment-1",
        "environment_revision": "environment-r1",
        "sample_frame_id": "world-frame",
        "sample_position_m": (0.0, 0.0, 0.0),
        "body_id": EMBODIMENT.body_id,
        "embodiment_episode_id": EMBODIMENT.embodiment_episode_id,
        "morphology_revision": EMBODIMENT.morphology_revision,
    }
    values.update(overrides)
    return SenseProvenance(**values)  # type: ignore[arg-type]


def _signal(provenance: SenseProvenance, value: float = 0.75) -> OpaqueSenseSignal:
    return OpaqueSenseSignal(
        channel_id="channel-1",
        receptor_id="receptor-1",
        value=value,
        provenance=provenance,
        contract_id=SCALAR_FIELD_CONTRACT_ID,
        contract_revision=SCALAR_FIELD_CONTRACT_REVISION,
        quantity_kind=SCALAR_FIELD_QUANTITY_KIND,
        unit=SCALAR_FIELD_UNIT,
    )


def _current() -> OrganismTransitionState:
    cognition_state = CognitiveTransitionState.initial(IDENTITY)
    runtime = CognitiveRuntimeCheckpoint(state=cognition_state, graph=_graph())
    owner_checkpoint = runtime.to_organism_checkpoint()
    return OrganismTransitionState(
        owner_checkpoint=owner_checkpoint,
        organism_step=4,
        embodiment=EMBODIMENT,
        world_correlation=WorldCausalCorrelation("world-1", 9),
        lifecycle=OrganismLifecycle.ACTIVE,
    )


def _advance(current: OrganismTransitionState, provenance: SenseProvenance):
    return advance_organism_from_sense_signals(
        current,
        (SenseChannelBinding("channel-1", "receptor-1", "sense.channel"),),
        (_signal(provenance),),
        expected_provenance=provenance,
        kernel_limits=LIMITS,
    )


def test_opaque_sense_and_cognition_are_accepted_as_one_checkpointed_step() -> None:
    current = _current()
    provenance = _provenance()

    result = _advance(current, provenance)

    assert current.organism_step == 4
    assert result.state.organism_step == 5
    assert result.state.identity == current.identity
    assert result.state.world_correlation == WorldCausalCorrelation("world-1", 10)
    assert result.composition.frame.activations["sense.channel"] == pytest.approx(0.75)
    assert current.owner_checkpoint != result.state.owner_checkpoint


def test_accepted_cognition_state_round_trips_with_transition_counter() -> None:
    accepted = _advance(_current(), _provenance()).state
    restored = OrganismTransitionState.from_bytes(accepted.to_bytes())
    runtime = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        restored.owner_checkpoint, kernel_limits=LIMITS
    )

    assert restored.organism_step == 5
    assert restored.world_correlation == WorldCausalCorrelation("world-1", 10)
    assert runtime.state.last_causal_tick == 10
    activations = dict(runtime.state.previous_activations)
    assert activations["private.state"] == pytest.approx(0.6351489523872873)
    assert activations["sense.channel"] == pytest.approx(0.75)


def test_accepted_transition_declares_no_external_owner_dependencies() -> None:
    result = _advance(_current(), _provenance())

    assert result.state.owner_checkpoint.dependencies == ()
    owner_payload = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        result.state.owner_checkpoint, kernel_limits=LIMITS
    )
    assert owner_payload.state.last_causal_tick == 10


def test_caller_dependencies_cannot_be_carried_into_next_owner_step() -> None:
    current = _current()
    forged = OrganismCheckpoint(
        current.identity,
        current.owner_checkpoint.private_state,
        (CheckpointDependency("models/unverified", "0" * 64),),
    )
    current = replace(current, owner_checkpoint=forged)
    before = current.to_bytes()

    with pytest.raises(CognitiveRuntimeCheckpointError, match="do not match"):
        _advance(current, _provenance())

    assert current.to_bytes() == before
    assert current.organism_step == 4


def test_mismatched_signal_is_rejected_without_publishing_or_incrementing() -> None:
    current = _current()
    before = current.to_bytes()
    expected = _provenance()
    stale = replace(expected, environment_revision="environment-old")

    with pytest.raises(ValueError, match="provenance"):
        advance_organism_from_sense_signals(
            current,
            (SenseChannelBinding("channel-1", "receptor-1", "sense.channel"),),
            (_signal(stale),),
            expected_provenance=expected,
            kernel_limits=LIMITS,
        )

    assert current.to_bytes() == before
    assert current.organism_step == 4


def test_wrong_embodiment_context_is_rejected_before_cognition() -> None:
    current = _current()
    before = current.to_bytes()
    wrong = _provenance(body_id="body-old")

    with pytest.raises(OrganismTransitionError, match="current Embodiment context"):
        _advance(current, wrong)

    assert current.to_bytes() == before


def test_stale_world_correlation_is_rejected_without_advancing() -> None:
    current = _current()
    stale = _provenance(causal_tick=9)

    with pytest.raises(OrganismTransitionError, match="must advance"):
        _advance(current, stale)

    assert current.organism_step == 4


def test_invalid_or_non_cognition_owner_checkpoint_fails_closed() -> None:
    current = _current()
    invalid = replace(
        current,
        owner_checkpoint=OrganismCheckpoint(IDENTITY, b"not a cognition runtime checkpoint"),
    )

    with pytest.raises(ValueError, match="malformed cognitive runtime checkpoint"):
        _advance(invalid, _provenance())

    assert invalid.organism_step == 4


def test_sequential_steps_advance_organism_step_and_world_correlation_monotonically() -> None:
    current = _current()
    assert current.organism_step == 4

    # Step 1: causal_tick 10 -> organism_step 5
    step1_prov = _provenance(causal_tick=10, physical_step_id="step-10")
    step1 = advance_organism_from_sense_signals(
        current,
        (SenseChannelBinding("channel-1", "receptor-1", "sense.channel"),),
        (_signal(step1_prov, value=0.5),),
        expected_provenance=step1_prov,
        kernel_limits=LIMITS,
    )
    assert step1.state.organism_step == 5
    assert step1.state.world_correlation == WorldCausalCorrelation("world-1", 10)

    # Step 2: causal_tick 11 -> organism_step 6
    step2_prov = _provenance(causal_tick=11, physical_step_id="step-11")
    step2 = advance_organism_from_sense_signals(
        step1.state,
        (SenseChannelBinding("channel-1", "receptor-1", "sense.channel"),),
        (_signal(step2_prov, value=0.8),),
        expected_provenance=step2_prov,
        kernel_limits=LIMITS,
    )
    assert step2.state.organism_step == 6
    assert step2.state.world_correlation == WorldCausalCorrelation("world-1", 11)

    # Step 3: verify unmutated earlier states and distinct checkpoints
    assert current.organism_step == 4
    assert step1.state.organism_step == 5
    assert step2.state.owner_checkpoint != step1.state.owner_checkpoint
    assert step1.state.owner_checkpoint != current.owner_checkpoint


def test_dormant_or_terminated_lifecycle_rejects_sense_progression() -> None:
    from organism.transition import transition_organism_lifecycle

    active = _current()
    dormant = transition_organism_lifecycle(
        active,
        OrganismLifecycle.DORMANT,
        transition_id="trans-dormant-1",
        request_ref="presence-loss",
    )
    before_bytes = dormant.to_bytes()

    with pytest.raises(OrganismTransitionError, match="requires an ACTIVE Organism"):
        _advance(dormant, _provenance())

    assert dormant.to_bytes() == before_bytes
    assert dormant.organism_step == active.organism_step
