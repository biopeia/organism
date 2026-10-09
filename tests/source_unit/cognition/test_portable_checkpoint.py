from __future__ import annotations

import json
from uuid import UUID

import pytest

from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.portable_checkpoint import (
    CognitiveRuntimeCheckpoint,
    CognitiveRuntimeCheckpointError,
)
from organism.cognition.transition import CognitiveTransitionState, advance_cognition
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity


def _graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=(PlasticNode("sense", NodeKind.SENSE), PlasticNode("memory", NodeKind.STATE)),
        edges=(
            PlasticEdge(
                source_id="sense",
                target_id="memory",
                kind=EdgeKind.EXCITATORY,
                weight=0.75,
                plasticity=0.1,
                delay_ticks=1,
                support=3,
                age_ticks=5,
                stable_ticks=2,
                last_use_tick=4,
            ),
        ),
        kernel_limits=KernelLimits(),
    )


def _state() -> CognitiveTransitionState:
    return CognitiveTransitionState.initial(
        OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000456"))
    )


def test_cognitive_graph_and_recurrent_state_round_trip_through_organism_envelope() -> None:
    state, _ = advance_cognition(_state(), _graph(), {"sense": 0.8}, causal_tick=12)
    original = CognitiveRuntimeCheckpoint(state=state, graph=_graph())
    envelope = original.to_organism_checkpoint()

    restored = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        type(envelope).from_bytes(envelope.to_bytes()), kernel_limits=KernelLimits()
    )

    assert restored.state == state
    assert [(node.node_id, node.kind) for node in restored.graph.nodes] == [
        (node.node_id, node.kind) for node in original.graph.nodes
    ]
    original_edge = original.graph.edges[0]
    restored_edge = restored.graph.edges[0]
    assert restored_edge.weight == original_edge.weight
    assert restored_edge.plasticity == original_edge.plasticity
    assert restored_edge.support == original_edge.support
    assert restored_edge.age_ticks == original_edge.age_ticks
    # Eligibility is deliberately labile and excluded by the existing graph codec.
    assert restored_edge.eligibility == 0.0


def test_cognitive_checkpoint_rejects_identity_mismatch() -> None:
    checkpoint = CognitiveRuntimeCheckpoint(_state(), _graph())
    document = json.loads(checkpoint.to_bytes())
    document["identity"] = str(
        OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000789"))
    )

    with pytest.raises(CognitiveRuntimeCheckpointError, match="identity mismatch"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


def test_cognitive_checkpoint_rejects_unknown_fields() -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["extra"] = True

    with pytest.raises(CognitiveRuntimeCheckpointError, match="shape"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )
