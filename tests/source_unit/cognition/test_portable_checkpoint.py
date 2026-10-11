from __future__ import annotations

import json
from hashlib import sha256
from uuid import UUID

import pytest

from organism.checkpoint import CheckpointDependency
from organism.cognition.activation import SensoryNormalizer
from organism.cognition.generative import (
    AgendaSource,
    GenerativeTarget,
    ResidentGenerativeCognition,
)
from organism.cognition.generative.adapters import EpisodicReplayAdapter
from organism.cognition.generative.registry import GenerativeModelRegistry
from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.metaplasticity import SafetyState
from organism.cognition.portable_checkpoint import (
    CognitiveRuntimeCheckpoint,
    CognitiveRuntimeCheckpointError,
)
from organism.cognition.transition import CognitiveTransitionState, advance_cognition
from organism.cognition.types import EdgeKind, NodeKind
from organism.genetics import (
    EpigeneticMark,
    GeneExpressionState,
    GermlineState,
    create_standard_genome,
)
from organism.identity import OrganismIdentity
from organism.memory import EpistemicStatus, ExperienceRecord, SourceKind


def _graph() -> CognitiveGraph:
    edge = PlasticEdge(
        source_id="sense",
        target_id="memory",
        kind=EdgeKind.EXCITATORY,
        weight=0.400001,
        plasticity=0.1,
        delay_ticks=1,
        eligibility=0.05,
        support=3,
        age_ticks=5,
        stable_ticks=2,
        last_use_tick=4,
    )
    return CognitiveGraph(
        nodes=(PlasticNode("sense", NodeKind.SENSE), PlasticNode("memory", NodeKind.STATE)),
        edges=(edge,),
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
    assert restored_edge.eligibility == original_edge.eligibility


def test_portable_checkpoint_composes_safety_and_normalizer_state() -> None:
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
    assert set(restored.sensory_normalizers) == {"established", "tentative"}
    assert restored.sensory_normalizers["established"].mean == pytest.approx(normalizer.mean)
    assert restored.sensory_normalizers["established"].variance == pytest.approx(
        normalizer.variance
    )
    assert restored.sensory_normalizers["established"].count == normalizer.count
    assert restored.sensory_normalizers["tentative"].count == tentative.count


def test_portable_checkpoint_rejects_malformed_safety_and_normalizer_payloads() -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["safety_state"] = {"consecutive_failures": -1, "frozen": True}
    with pytest.raises(CognitiveRuntimeCheckpointError, match="malformed cognitive runtime"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )

    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["sensory_normalizers"] = {
        "bad": {"mean": 1.0, "variance": -1.0, "count": 1}
    }
    with pytest.raises(CognitiveRuntimeCheckpointError, match="malformed cognitive runtime"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )

    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["sensory_normalizers"] = None
    with pytest.raises(CognitiveRuntimeCheckpointError, match="malformed cognitive runtime"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )

    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["sensory_normalizers"] = {
        "huge": {"mean": 10**1000, "variance": 1.0, "count": 1}
    }
    with pytest.raises(CognitiveRuntimeCheckpointError, match="malformed cognitive runtime"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


def test_portable_checkpoint_preserves_unestablished_normalizer_continuation() -> None:
    normalizer = SensoryNormalizer()
    normalizer.normalize(1.0)
    checkpoint = CognitiveRuntimeCheckpoint(
        _state(),
        _graph(),
        sensory_normalizers={"sense": normalizer, "fresh": SensoryNormalizer()},
    )

    restored = CognitiveRuntimeCheckpoint.from_bytes(
        checkpoint.to_bytes(), kernel_limits=KernelLimits()
    )

    restored_normalizer = restored.sensory_normalizers["sense"]
    assert restored.sensory_normalizers["fresh"].count == 0
    assert restored_normalizer.mean == pytest.approx(normalizer.mean)
    assert restored_normalizer.variance == pytest.approx(normalizer.variance)
    assert restored_normalizer.count == normalizer.count
    for value in (1.5, -0.25, 2.0):
        assert restored_normalizer.normalize(value) == pytest.approx(
            normalizer.normalize(value)
        )


@pytest.mark.parametrize("prior_version", [6, 7])
def test_prior_portable_checkpoint_schemas_are_not_accepted(prior_version: int) -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["schema_version"] = prior_version
    with pytest.raises(CognitiveRuntimeCheckpointError, match="unsupported"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


@pytest.mark.parametrize("invalid_version", [8.0, True])
def test_portable_checkpoint_rejects_non_integer_schema_version(invalid_version: object) -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["schema_version"] = invalid_version

    with pytest.raises(CognitiveRuntimeCheckpointError, match="unsupported"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


def _genetic_state():
    genome = create_standard_genome("portable-genetic-checkpoint")
    inherited = EpigeneticMark("plasticity.learning_rate.baseline", delta=0.01, strength=0.5)
    germline = GermlineState.from_genome(
        genome, inherited_marks=(inherited,), acquired_capture_enabled=True
    )
    acquired = EpigeneticMark("structure.growth_threshold.baseline", delta=0.02)
    germline.acquired_marks[acquired.locus] = acquired
    expression = GeneExpressionState.from_genome(genome, germline=germline)
    return genome, germline, expression


def test_portable_checkpoint_round_trips_genome_germline_and_expression_as_one_unit() -> None:
    genome, germline, expression = _genetic_state()
    original = CognitiveRuntimeCheckpoint(
        state=_state(),
        graph=_graph(),
        genome=genome,
        germline_state=germline,
        expression_state=expression,
    )

    restored = CognitiveRuntimeCheckpoint.from_bytes(
        original.to_bytes(),
        kernel_limits=KernelLimits(),
        running_version=(0, 85, 0),
    )

    assert restored.genome == genome
    assert restored.genome is not None
    assert restored.genome.genome_hash == genome.genome_hash
    assert restored.germline_state is not None
    assert restored.germline_state.birth_expression == germline.birth_expression
    assert restored.germline_state.inherited_marks == germline.inherited_marks
    assert restored.germline_state.acquired_marks == germline.acquired_marks
    assert restored.expression_state is not None
    assert restored.expression_state.as_dict() == expression.as_dict()


@pytest.mark.parametrize(
    "provided",
    [
        {"genome": create_standard_genome("partial-genome")},
        {
            "genome": create_standard_genome("partial-germline"),
            "germline_state": GermlineState.from_genome(
                create_standard_genome("partial-germline")
            ),
        },
    ],
)
def test_portable_checkpoint_rejects_partial_genetic_state(provided) -> None:
    with pytest.raises(CognitiveRuntimeCheckpointError, match="supplied together"):
        CognitiveRuntimeCheckpoint(state=_state(), graph=_graph(), **provided)


def test_portable_checkpoint_genetic_restore_requires_explicit_running_version() -> None:
    genome, germline, expression = _genetic_state()
    checkpoint = CognitiveRuntimeCheckpoint(
        state=_state(),
        graph=_graph(),
        genome=genome,
        germline_state=germline,
        expression_state=expression,
    )

    with pytest.raises(CognitiveRuntimeCheckpointError, match="running_version is required"):
        CognitiveRuntimeCheckpoint.from_bytes(checkpoint.to_bytes(), kernel_limits=KernelLimits())


def test_portable_checkpoint_rejects_tampered_genetic_state() -> None:
    genome, germline, expression = _genetic_state()
    checkpoint = CognitiveRuntimeCheckpoint(
        state=_state(),
        graph=_graph(),
        genome=genome,
        germline_state=germline,
        expression_state=expression,
    )
    document = json.loads(checkpoint.to_bytes())
    document["genome"]["genotype_hash"] = "0" * len(genome.genotype_hash)

    with pytest.raises(CognitiveRuntimeCheckpointError, match="malformed cognitive runtime"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(),
            kernel_limits=KernelLimits(),
            running_version=(0, 85, 0),
        )


def test_portable_checkpoint_rejects_inconsistent_genetic_state_at_construction() -> None:
    genome, germline, expression = _genetic_state()
    expression.birth_expression["plasticity.learning_rate.baseline"] += 0.1

    with pytest.raises(CognitiveRuntimeCheckpointError, match="genetic state is inconsistent"):
        CognitiveRuntimeCheckpoint(
            state=_state(),
            graph=_graph(),
            genome=genome,
            germline_state=germline,
            expression_state=expression,
        )


def test_portable_checkpoint_rejects_genetic_state_mutated_after_construction() -> None:
    genome, germline, expression = _genetic_state()
    checkpoint = CognitiveRuntimeCheckpoint(
        state=_state(),
        graph=_graph(),
        genome=genome,
        germline_state=germline,
        expression_state=expression,
    )
    # The checkpoint is frozen, but nested domain values are mutable. The
    # serialized snapshot must therefore be revalidated, not trusted from
    # construction time.
    expression.birth_expression["plasticity.learning_rate.baseline"] += 0.1

    with pytest.raises(CognitiveRuntimeCheckpointError, match="genetic state is inconsistent"):
        checkpoint.to_bytes()


def test_portable_checkpoint_rejects_serialized_partial_genetic_state() -> None:
    genome, germline, expression = _genetic_state()
    document = json.loads(
        CognitiveRuntimeCheckpoint(
            state=_state(),
            graph=_graph(),
            genome=genome,
            germline_state=germline,
            expression_state=expression,
        ).to_bytes()
    )
    document["expression_state"] = None

    with pytest.raises(CognitiveRuntimeCheckpointError, match="present together"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(),
            kernel_limits=KernelLimits(),
            running_version=(0, 85, 0),
        )


def test_portable_checkpoint_round_trips_genetics_through_organism_envelope() -> None:
    genome, germline, expression = _genetic_state()
    envelope = CognitiveRuntimeCheckpoint(
        state=_state(),
        graph=_graph(),
        genome=genome,
        germline_state=germline,
        expression_state=expression,
    ).to_organism_checkpoint()

    restored = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        envelope,
        kernel_limits=KernelLimits(),
        running_version=(0, 85, 0),
    )

    assert restored.genome == genome
    assert restored.germline_state is not None
    assert restored.germline_state.inherited_marks == germline.inherited_marks
    assert restored.expression_state is not None
    assert restored.expression_state.as_dict() == expression.as_dict()


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


def test_portable_checkpoint_round_trips_explicit_generative_resident_state() -> None:
    resident = ResidentGenerativeCognition(organism_id=str(_state().identity))
    resident.generative_tick = 9
    resident.agenda.add_target(
        GenerativeTarget(
            target_id="target.longitudinal",
            source=AgendaSource.UNCERTAINTY,
            source_refs=("organism.signal.1",),
            created_tick=4,
            uncertainty=0.7,
            persistence=0.5,
            recurrence=2,
        )
    )
    before = CognitiveRuntimeCheckpoint(_state(), _graph(), generative_resident=resident)

    restored = CognitiveRuntimeCheckpoint.from_bytes(
        before.to_bytes(), kernel_limits=KernelLimits()
    )

    assert restored.generative_resident is not None
    assert restored.generative_resident.organism_id == str(_state().identity)
    assert restored.generative_resident.generative_tick == 9
    assert restored.generative_resident.agenda.targets == resident.agenda.targets


def test_portable_checkpoint_restores_with_explicit_empty_model_registry() -> None:
    resident = ResidentGenerativeCognition(organism_id=str(_state().identity))
    registry = GenerativeModelRegistry()
    payload = CognitiveRuntimeCheckpoint(
        _state(), _graph(), generative_resident=resident
    ).to_bytes()

    restored = CognitiveRuntimeCheckpoint.from_bytes(
        payload, kernel_limits=KernelLimits(), generative_registry=registry
    )

    assert restored.generative_resident is not None
    assert restored.generative_resident.registry is registry


def test_portable_checkpoint_rejects_generative_state_for_another_organism() -> None:
    resident = ResidentGenerativeCognition(organism_id="organism.other")

    with pytest.raises(CognitiveRuntimeCheckpointError, match="different Organism identity"):
        CognitiveRuntimeCheckpoint(_state(), _graph(), generative_resident=resident)


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


def test_owner_payload_rejects_caller_declared_dependencies() -> None:
    artifact = b"private learned model fixture"
    checkpoint = CognitiveRuntimeCheckpoint(_state(), _graph()).to_organism_checkpoint()
    assert checkpoint.dependencies == ()
    extra = type(checkpoint)(
        checkpoint.identity,
        checkpoint.private_state,
        (CheckpointDependency("models/model-1", sha256(artifact).hexdigest()),),
    )

    with pytest.raises(CognitiveRuntimeCheckpointError, match="do not match"):
        CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            extra, kernel_limits=KernelLimits()
        )
    restored = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        checkpoint, kernel_limits=KernelLimits()
    )
    assert restored.state.identity == checkpoint.identity


def test_owner_payload_rejects_undeclared_external_generative_state() -> None:
    resident = ResidentGenerativeCognition(organism_id=str(_state().identity))
    resident.register_model(EpisodicReplayAdapter("model-1", projector=lambda _context: ()))
    runtime = CognitiveRuntimeCheckpoint(_state(), _graph(), generative_resident=resident)
    with pytest.raises(CognitiveRuntimeCheckpointError, match="external generative dependencies"):
        runtime.to_organism_checkpoint()

    resident.registry.unregister("model-1")
    resident.set_factual_evidence_resolver(lambda _ref: True)
    with pytest.raises(CognitiveRuntimeCheckpointError, match="external generative dependencies"):
        runtime.to_bytes()


def test_owner_payload_rejects_injected_registry() -> None:
    resident = ResidentGenerativeCognition(organism_id=str(_state().identity))
    runtime = CognitiveRuntimeCheckpoint(_state(), _graph(), generative_resident=resident)
    checkpoint = runtime.to_organism_checkpoint()
    registry = GenerativeModelRegistry()
    registry.register(EpisodicReplayAdapter("model-1", projector=lambda _context: ()))
    with pytest.raises(CognitiveRuntimeCheckpointError, match="external generative dependencies"):
        CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            checkpoint, kernel_limits=KernelLimits(), generative_registry=registry
        )


@pytest.mark.parametrize("invalid_closure", [None, False, 0, "", {}, [1], [{"locator": "x"}]])
def test_owner_payload_rejects_unsupported_dependency_manifest(invalid_closure: object) -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())
    document["dependency_closure"] = invalid_closure
    with pytest.raises(CognitiveRuntimeCheckpointError, match="unsupported external dependency"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )


def test_owner_payload_requires_dependency_manifest() -> None:
    document = json.loads(CognitiveRuntimeCheckpoint(_state(), _graph()).to_bytes())

    del document["dependency_closure"]
    with pytest.raises(CognitiveRuntimeCheckpointError, match="shape"):
        CognitiveRuntimeCheckpoint.from_bytes(
            json.dumps(document).encode(), kernel_limits=KernelLimits()
        )
