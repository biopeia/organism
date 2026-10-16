"""Differential and stability test suite for unsupervised plasticity in Organism.

Verifies:
1. Null stimulus control (x = 0): zero spontaneous drift.
2. Orthogonal channel isolation: distractor activation does not alter target plastic edge.
3. Stability across varied initial weights w_0 and stimuli intensities.
4. Memory retention: weight is preserved when stimulus ceases.
5. Functional differential: probe stimulus produces subthreshold response in control
   and suprathreshold response in plastic organism.
6. Longitudinal survival: learned weight and functional consequence survive cold serialization
   round-trip and allow smooth continuation.
"""

from __future__ import annotations

import math
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
from organism.cognition import (
    CognitiveRuntimeCheckpoint,
    CognitiveTransitionState,
    EdgeLearningInstruction,
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    advance_organism_from_sense_signals,
)
from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.sense_input import (
    SCALAR_FIELD_CONTRACT_ID,
    SCALAR_FIELD_CONTRACT_REVISION,
    SCALAR_FIELD_QUANTITY_KIND,
    SCALAR_FIELD_UNIT,
)
from organism.cognition.types import EdgeKind, NodeKind

IDENTITY = OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000555"))
LIMITS = KernelLimits()
EMBODIMENT = EmbodimentContextRef("body-1", "episode-1", "morph-1")


def _provenance(causal_tick: int) -> SenseProvenance:
    return SenseProvenance(
        world_id="world-test",
        causal_tick=causal_tick,
        physical_step_id=f"step-{causal_tick}",
        environment_id="env-1",
        environment_revision="env-r1",
        sample_frame_id="frame-1",
        sample_position_m=(0.0, 0.0, 0.0),
        body_id=EMBODIMENT.body_id,
        embodiment_episode_id=EMBODIMENT.embodiment_episode_id,
        morphology_revision=EMBODIMENT.morphology_revision,
    )


def _signal(provenance: SenseProvenance, channel_id: str, value: float) -> OpaqueSenseSignal:
    return OpaqueSenseSignal(
        channel_id=channel_id,
        receptor_id=f"receptor.{channel_id}",
        value=value,
        provenance=provenance,
        contract_id=SCALAR_FIELD_CONTRACT_ID,
        contract_revision=SCALAR_FIELD_CONTRACT_REVISION,
        quantity_kind=SCALAR_FIELD_QUANTITY_KIND,
        unit=SCALAR_FIELD_UNIT,
    )


def _create_organism(
    initial_weight: float = 0.20,
    with_distractor: bool = False,
) -> tuple[OrganismTransitionState, tuple[SenseChannelBinding, ...]]:
    nodes = [
        PlasticNode("sense.target", NodeKind.SENSE),
        PlasticNode("concept.target", NodeKind.CONCEPT),
    ]
    edges = [
        PlasticEdge(
            "sense.target", "concept.target", EdgeKind.EXCITATORY,
            initial_weight, 0.0, 0,
        ),
    ]
    bindings = [
        SenseChannelBinding("ch.target", "receptor.ch.target", "sense.target"),
    ]

    if with_distractor:
        nodes.append(PlasticNode("sense.distractor", NodeKind.SENSE))
        nodes.append(PlasticNode("concept.distractor", NodeKind.CONCEPT))
        edges.append(
            PlasticEdge(
                "sense.distractor", "concept.distractor", EdgeKind.EXCITATORY,
                0.20, 0.0, 0,
            )
        )
        bindings.append(
            SenseChannelBinding("ch.distractor", "receptor.ch.distractor", "sense.distractor")
        )

    graph = CognitiveGraph(nodes=tuple(nodes), edges=tuple(edges), kernel_limits=LIMITS)
    cognition = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState.initial(IDENTITY),
        graph=graph,
    )
    organism = OrganismTransitionState(
        owner_checkpoint=cognition.to_organism_checkpoint(),
        organism_step=0,
        embodiment=EMBODIMENT,
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    return organism, tuple(bindings)


def _run_ticks(
    organism: OrganismTransitionState,
    bindings: tuple[SenseChannelBinding, ...],
    ticks: int,
    signal_values: dict[str, float],
    learning: tuple[EdgeLearningInstruction, ...],
    start_tick: int = 1,
) -> OrganismTransitionState:
    current = organism
    for tick in range(start_tick, start_tick + ticks):
        prov = _provenance(tick)
        signals = tuple(_signal(prov, ch_id, val) for ch_id, val in signal_values.items())
        res = advance_organism_from_sense_signals(
            current,
            bindings=bindings,
            signals=signals,
            expected_provenance=prov,
            kernel_limits=LIMITS,
            learning=learning,
        )
        current = res.state
    return current


def _get_weight(organism: OrganismTransitionState, source_id: str, target_id: str) -> float:
    chk = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        organism.owner_checkpoint, kernel_limits=LIMITS
    )
    for edge in chk.graph.edges:
        if edge.source_id == source_id and edge.target_id == target_id:
            return edge.weight
    raise KeyError(f"edge {source_id} -> {target_id} not found")


def test_null_stimulus_control_produces_zero_weight_drift() -> None:
    """When input is zero, the Oja weight update must be exactly zero (no spontaneous drift)."""
    organism, bindings = _create_organism(initial_weight=0.20)
    learning = (
        EdgeLearningInstruction(
            source_id="sense.target",
            target_id="concept.target",
            kind=EdgeKind.EXCITATORY,
            learning_rate=0.10,
            modulation=1.0,
            eligibility_decay=0.90,
            eligible=True,
        ),
    )

    result = _run_ticks(
        organism,
        bindings,
        ticks=30,
        signal_values={"ch.target": 0.0},
        learning=learning,
    )
    w_end = _get_weight(result, "sense.target", "concept.target")
    assert w_end == 0.20, f"Weight must remain exactly 0.20 under null stimulus, got {w_end}"


def test_orthogonal_channel_isolation() -> None:
    """Activating an unlinked distractor channel must not affect the target edge weight."""
    organism, bindings = _create_organism(initial_weight=0.20, with_distractor=True)
    learning = (
        EdgeLearningInstruction(
            source_id="sense.target",
            target_id="concept.target",
            kind=EdgeKind.EXCITATORY,
            learning_rate=0.10,
            modulation=1.0,
            eligibility_decay=0.90,
            eligible=True,
        ),
    )

    # Distractor active (0.90), target inactive (0.0)
    result = _run_ticks(
        organism,
        bindings,
        ticks=25,
        signal_values={"ch.target": 0.0, "ch.distractor": 0.90},
        learning=learning,
    )
    w_target = _get_weight(result, "sense.target", "concept.target")
    assert w_target == 0.20, f"Target weight must stay 0.20 despite distractor activation, got {w_target}"


@pytest.mark.parametrize("initial_weight", [0.05, 0.20, 0.50, 0.80])
@pytest.mark.parametrize("stimulus", [0.20, 0.50, 0.90])
def test_oja_plasticity_monotonic_convergence(initial_weight: float, stimulus: float) -> None:
    """Oja rule converges stably towards its theoretical fixed point without numerical oscillation."""
    organism, bindings = _create_organism(initial_weight=initial_weight)
    learning = (
        EdgeLearningInstruction(
            source_id="sense.target",
            target_id="concept.target",
            kind=EdgeKind.EXCITATORY,
            learning_rate=0.08,
            modulation=1.0,
            eligibility_decay=0.90,
            eligible=True,
        ),
    )

    weights: list[float] = [initial_weight]
    current = organism
    for tick in range(1, 41):
        prov = _provenance(tick)
        signals = (_signal(prov, "ch.target", stimulus),)
        res = advance_organism_from_sense_signals(
            current,
            bindings=bindings,
            signals=signals,
            expected_provenance=prov,
            kernel_limits=LIMITS,
            learning=learning,
        )
        current = res.state
        weights.append(_get_weight(current, "sense.target", "concept.target"))

    # Monotonic progression towards fixed point (fixed point is ~1.0 - 1.14 for these stimuli)
    for i in range(1, len(weights)):
        assert weights[i] > weights[i - 1], (
            f"Weight must strictly increase monotonically at tick {i}: {weights[i]} <= {weights[i-1]}"
        )
    assert all(math.isfinite(w) for w in weights), "Weights must remain finite"
    assert all(0.0 <= w <= 2.0 for w in weights), "Weights must remain bounded within range"


def test_memory_retention_when_stimulus_ceases() -> None:
    """Learned weight is retained when training stimulus is removed."""
    organism, bindings = _create_organism(initial_weight=0.20)
    learning = (
        EdgeLearningInstruction(
            source_id="sense.target",
            target_id="concept.target",
            kind=EdgeKind.EXCITATORY,
            learning_rate=0.10,
            modulation=1.0,
            eligibility_decay=0.90,
            eligible=True,
        ),
    )

    # Phase 1: Training (20 ticks with stimulus 0.85)
    trained = _run_ticks(
        organism, bindings, ticks=20,
        signal_values={"ch.target": 0.85},
        learning=learning,
        start_tick=1,
    )
    w_trained = _get_weight(trained, "sense.target", "concept.target")
    assert w_trained > 0.50, f"Expected potentiation > 0.50, got {w_trained}"

    # Phase 2: Rest / Extinction (20 ticks with stimulus 0.0)
    rested = _run_ticks(
        trained, bindings, ticks=20,
        signal_values={"ch.target": 0.0},
        learning=learning,
        start_tick=21,
    )
    w_rested = _get_weight(rested, "sense.target", "concept.target")
    assert w_rested == w_trained, f"Memory must be retained exactly: {w_rested} != {w_trained}"


def test_functional_consequence_and_cold_restoration_continuity() -> None:
    """Trained organism demonstrates functional difference that survives cold restore and continues learning."""
    organism, bindings = _create_organism(initial_weight=0.20)
    learning = (
        EdgeLearningInstruction(
            source_id="sense.target",
            target_id="concept.target",
            kind=EdgeKind.EXCITATORY,
            learning_rate=0.10,
            modulation=1.0,
            eligibility_decay=0.90,
            eligible=True,
        ),
    )

    # Control run (learning disabled)
    control = _run_ticks(
        organism, bindings, ticks=20,
        signal_values={"ch.target": 0.85},
        learning=(),
        start_tick=1,
    )
    w_control = _get_weight(control, "sense.target", "concept.target")
    assert w_control == 0.20

    # Plastic run (learning enabled)
    plastic = _run_ticks(
        organism, bindings, ticks=20,
        signal_values={"ch.target": 0.85},
        learning=learning,
        start_tick=1,
    )
    w_plastic = _get_weight(plastic, "sense.target", "concept.target")
    assert w_plastic > 0.60

    # Functional evaluation under weak probe input x = 0.40
    probe_x = 0.40
    threshold = 0.20
    activation_control = math.tanh(w_control * probe_x)
    activation_plastic = math.tanh(w_plastic * probe_x)

    assert activation_control < threshold, f"Control must be subthreshold: {activation_control} < {threshold}"
    assert activation_plastic >= threshold, f"Plastic must be suprathreshold: {activation_plastic} >= {threshold}"

    # Cold restoration from serialized bytes
    raw_bytes = plastic.owner_checkpoint.to_bytes()
    restored_chk = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        OrganismCheckpoint.from_bytes(raw_bytes), kernel_limits=LIMITS
    )
    restored_organism = OrganismTransitionState(
        owner_checkpoint=restored_chk.to_organism_checkpoint(),
        organism_step=plastic.organism_step,
        embodiment=plastic.embodiment,
        world_correlation=plastic.world_correlation,
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    w_restored = _get_weight(restored_organism, "sense.target", "concept.target")
    assert w_restored == w_plastic

    activation_restored = math.tanh(w_restored * probe_x)
    assert activation_restored == activation_plastic

    # Post-restoration continuation
    resumed = _run_ticks(
        restored_organism, bindings, ticks=10,
        signal_values={"ch.target": 0.85},
        learning=learning,
        start_tick=21,
    )
    w_resumed = _get_weight(resumed, "sense.target", "concept.target")
    assert w_resumed > w_restored, "Continued learning post-restoration must advance weight"
