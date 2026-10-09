from __future__ import annotations

from hashlib import sha256
import json
from uuid import UUID

import pytest

from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.checkpoint import CheckpointDependency
from organism.cognition.limits import KernelLimits
from organism.cognition.portable_checkpoint import (
    CognitiveRuntimeCheckpoint,
    CognitiveRuntimeCheckpointError,
)
from organism.cognition.activation import SensoryNormalizer
from organism.cognition.metaplasticity import SafetyState
from organism.cognition.transition import CognitiveTransitionState, advance_cognition
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity
from organism.memory import EpistemicStatus, ExperienceRecord, SourceKind


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


def _experience_record(*, organism_id: str | None = None) -> ExperienceRecord:
    return ExperienceRecord(
        record_id="experience.1",
        organism_id=organism_id or str(_state().identity),
        tick_class=12,
        context_tokens=("signal.opaque.1",),
        action_token="action.opaque.1",
        outcome_tokens=("outcome.opaque.1",),
        epistemic_status=EpistemicStatus.OBSERVED,
        evidence_refs=("causal-record.12",),
        confidence_class=6,
        source_kind=SourceKind.DIRECT,
    )


def test_cognitive_graph_and_recurrent_state_round_trip_through_organism_envelope() -> None:
    state, _ = advance_cognition(_state(), _graph(), {"sense": 0.8}, causal_tick=12)
    record = _experience_record(organism_id=str(state.identity))
    original = CognitiveRuntimeCheckpoint(
        state=state, graph=_graph(), experience_records=(record,)
    )
    envelope = original.to_organism_checkpoint()

    restored = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        type(envelope).from_bytes(envelope.to_bytes()), kernel_limits=KernelLimits()
    )

    assert restored.state == state
    assert restored.experience_records == (record,)
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


def test_portable_checkpoint_composes_safety_and_established_normalizer_state() -> None:
    safety = SafetyState()
    for _ in range(3):
        safety.record_failure()
    normalizer = SensoryNormalizer()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        normalizer.normalize(value)
    tentative = SensoryNormalizer()
    tentative.normalize(9.0)

    original = CognitiveRuntimeCheckpoint(
        state=_state(),
        graph=_graph(),
        safety_state=safety,
        sensory_normalizers={"established": normalizer, "tentative": tentative},
    )
    restored = CognitiveRuntimeCheckpoint.from_bytes(
        original.to_bytes(), kernel_limits=KernelLimits()
    )

    assert restored.safety_state == safety
    assert set(restored.sensory_normalizers) == {"established"}
    assert restored.sensory_normalizers["established"].mean == pytest.approx(normalizer.mean)
    assert restored.sensory_normalizers["established"].variance == pytest.approx(
        normalizer.variance
    )
    assert restored.sensory_normalizers["established"].count == normalizer.count


def test_portable_checkpoint_rejects_malformed_safety_and_normalizer_payloads() -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["safety_state"] = {"consecutive_failures": -1, "frozen": True}
    with pytest.raises(CognitiveRuntimeCheckpointError, match="malformed cognitive runtime"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )

    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["sensory_normalizers"] = {
        "bad": {"mean": 1.0, "variance": 1.0, "count": 1}
    }
    with pytest.raises(CognitiveRuntimeCheckpointError, match="malformed cognitive runtime"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


def test_portable_checkpoint_schema_v2_is_not_accepted_as_a_current_payload() -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["schema_version"] = 2
    with pytest.raises(CognitiveRuntimeCheckpointError, match="unsupported"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


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


def test_cognitive_checkpoint_rejects_experience_from_another_identity() -> None:
    with pytest.raises(CognitiveRuntimeCheckpointError, match="different Organism identity"):
        CognitiveRuntimeCheckpoint(
            state=_state(),
            graph=_graph(),
            experience_records=(_experience_record(organism_id="other"),),
        )


def test_cognitive_checkpoint_rejects_duplicate_experience_record_identity() -> None:
    record = _experience_record()
    with pytest.raises(CognitiveRuntimeCheckpointError, match="duplicate experience"):
        CognitiveRuntimeCheckpoint(
            state=_state(), graph=_graph(), experience_records=(record, record)
        )


def test_cognitive_checkpoint_requires_experience_records_in_current_schema() -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    del document["experience_records"]

    with pytest.raises(CognitiveRuntimeCheckpointError, match="shape"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


def test_organism_checkpoint_restore_requires_and_verifies_dependency_closure() -> None:
    artifact = b"private learned model fixture"
    checkpoint = CognitiveRuntimeCheckpoint(_state(), _graph()).to_organism_checkpoint(
        dependencies=(
            CheckpointDependency("models/model-1", sha256(artifact).hexdigest()),
        )
    )

    with pytest.raises(CognitiveRuntimeCheckpointError, match="resolver is required"):
        CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            checkpoint, kernel_limits=KernelLimits()
        )

    def missing_dependency(_locator: str) -> bytes:
        raise FileNotFoundError("not present")

    with pytest.raises(CognitiveRuntimeCheckpointError, match="closure is invalid"):
        CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            checkpoint,
            kernel_limits=KernelLimits(),
            read_dependency=missing_dependency,
        )

    with pytest.raises(CognitiveRuntimeCheckpointError, match="closure is invalid"):
        CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            checkpoint,
            kernel_limits=KernelLimits(),
            read_dependency=lambda _locator: b"wrong content",
        )

    restored = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        checkpoint,
        kernel_limits=KernelLimits(),
        read_dependency=lambda locator: artifact if locator == "models/model-1" else b"",
    )
    assert restored.state.identity == checkpoint.identity
    assert restored.graph is not None
