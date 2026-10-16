from __future__ import annotations

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
from organism.memory import (
    EpistemicStatus,
    ExperienceRecordingPolicy,
    SourceKind,
    accumulate_experience_records,
    record_action_outcome,
    record_cognitive_interpretation,
    record_sense_reception,
)

IDENTITY = OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000123"))
LIMITS = KernelLimits()
EMBODIMENT = EmbodimentContextRef("body-1", "episode-2", "morph-7")


def _graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=(
            PlasticNode("sense.channel", NodeKind.SENSE),
            PlasticNode("concept.alert", NodeKind.CONCEPT),
        ),
        edges=(
            PlasticEdge(
                "sense.channel", "concept.alert", EdgeKind.EXCITATORY,
                1.0, 0.0, 0,
            ),
        ),
        kernel_limits=LIMITS,
    )


def _active_state() -> OrganismTransitionState:
    cognition_state = CognitiveTransitionState.initial(IDENTITY)
    runtime = CognitiveRuntimeCheckpoint(state=cognition_state, graph=_graph())
    return OrganismTransitionState(
        owner_checkpoint=runtime.to_organism_checkpoint(),
        organism_step=1,
        embodiment=EMBODIMENT,
        world_correlation=WorldCausalCorrelation("world-1", 3),
        lifecycle=OrganismLifecycle.ACTIVE,
    )


def _provenance(causal_tick: int = 10, **overrides: object) -> SenseProvenance:
    values: dict[str, object] = {
        "world_id": "world-1",
        "causal_tick": causal_tick,
        "physical_step_id": f"step-{causal_tick}",
        "environment_id": "env-1",
        "environment_revision": "env-r1",
        "sample_frame_id": "frame-1",
        "sample_position_m": (0.0, 0.0, 0.0),
        "body_id": EMBODIMENT.body_id,
        "embodiment_episode_id": EMBODIMENT.embodiment_episode_id,
        "morphology_revision": EMBODIMENT.morphology_revision,
    }
    values.update(overrides)
    return SenseProvenance(**values)  # type: ignore[arg-type]


def _signal(provenance: SenseProvenance, channel_id: str = "ch-1", value: float = 0.75) -> OpaqueSenseSignal:
    return OpaqueSenseSignal(
        channel_id=channel_id,
        receptor_id="receptor-1",
        value=value,
        provenance=provenance,
        contract_id=SCALAR_FIELD_CONTRACT_ID,
        contract_revision=SCALAR_FIELD_CONTRACT_REVISION,
        quantity_kind=SCALAR_FIELD_QUANTITY_KIND,
        unit=SCALAR_FIELD_UNIT,
    )


def test_experience_recording_policy_validation() -> None:
    policy = ExperienceRecordingPolicy(max_records=64, activation_threshold=0.3)
    assert policy.max_records == 64
    assert policy.activation_threshold == 0.3
    assert policy.record_reception is True

    with pytest.raises(ValueError, match="max_records must be an integer within"):
        ExperienceRecordingPolicy(max_records=10)

    with pytest.raises(ValueError, match="activation_threshold must be a float"):
        ExperienceRecordingPolicy(activation_threshold=1.5)

    with pytest.raises(ValueError, match="evidence_prefix must be a bounded non-empty ASCII"):
        ExperienceRecordingPolicy(evidence_prefix="invalid prefix with space")


def test_record_sense_reception_creates_direct_observed_record() -> None:
    prov = _provenance(10)
    signals = (_signal(prov, "ch1", 0.75), _signal(prov, "ch2", 0.25))

    record = record_sense_reception("org-1", 10, signals, prov)

    assert record.organism_id == "org-1"
    assert record.tick_class == 10
    assert record.source_kind is SourceKind.DIRECT
    assert record.epistemic_status is EpistemicStatus.OBSERVED
    assert record.evidence_refs == ("sense:world-1:10",)
    assert record.confidence_class == 7
    assert record.action_token is None
    assert "ch:ch1:0.7500" in record.context_tokens
    assert "ch:ch2:0.2500" in record.context_tokens


def test_record_cognitive_interpretation_creates_cognitive_associated_record() -> None:
    activations = {"node.a": 0.85, "node.b": 0.12, "node.c": -0.65}
    record = record_cognitive_interpretation("org-1", 10, activations, threshold=0.5)

    assert record is not None
    assert record.organism_id == "org-1"
    assert record.source_kind is SourceKind.COGNITIVE
    assert record.epistemic_status is EpistemicStatus.ASSOCIATED
    assert record.evidence_refs == ()
    assert record.confidence_class == 4
    assert len(record.context_tokens) == 2  # node.a and node.c
    assert any("node.a" in t for t in record.context_tokens)
    assert any("node.c" in t for t in record.context_tokens)

    # Sub-threshold returns None
    assert record_cognitive_interpretation("org-1", 10, activations, threshold=0.99) is None


def test_record_action_outcome_creates_knowledge_record() -> None:
    record = record_action_outcome(
        "org-1",
        12,
        action_token="actuator_0",
        outcome_tokens=("delta_pos:0.15",),
        evidence_ref="receipt:world-1:12",
        supported=True,
    )
    assert record.source_kind is SourceKind.ACTION_OUTCOME
    assert record.epistemic_status is EpistemicStatus.SUPPORTED
    assert record.evidence_refs == ("receipt:world-1:12",)
    assert record.action_token == "actuator_0"
    assert record.confidence_class == 6

    # Contradicted outcome
    contra = record_action_outcome(
        "org-1",
        13,
        action_token="actuator_0",
        outcome_tokens=("delta_pos:0.0",),
        evidence_ref="receipt:world-1:13",
        supported=False,
    )
    assert contra.epistemic_status is EpistemicStatus.CONTRADICTED


def test_accumulate_experience_records_fifo_eviction() -> None:
    records = [
        record_sense_reception("org-1", i, (), _provenance(i))
        for i in range(20)
    ]
    accumulated = accumulate_experience_records("org-1", (), records, max_records=16)
    assert len(accumulated) == 16
    # Oldest (ticks 0..3) evicted, ticks 4..19 retained
    assert accumulated[0].tick_class == 4
    assert accumulated[-1].tick_class == 19


def test_advance_organism_records_experience_opt_in() -> None:
    current = _active_state()
    binding = SenseChannelBinding("ch-0", "receptor-1", "sense.channel")
    prov = _provenance(4)
    signal = _signal(prov, channel_id="ch-0", value=0.8)

    # 1. Default: experience_policy is None -> experience_records remains empty
    res_default = advance_organism_from_sense_signals(
        current,
        bindings=(binding,),
        signals=(signal,),
        expected_provenance=prov,
        kernel_limits=LIMITS,
    )
    runtime_default = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        res_default.state.owner_checkpoint, kernel_limits=LIMITS
    )
    assert len(runtime_default.experience_records) == 0

    # 2. Opt-in: experience_policy enabled for reception and interpretation
    policy = ExperienceRecordingPolicy(
        record_reception=True,
        record_interpretation=True,
        activation_threshold=0.1,
    )
    res_recorded = advance_organism_from_sense_signals(
        current,
        bindings=(binding,),
        signals=(signal,),
        expected_provenance=prov,
        kernel_limits=LIMITS,
        experience_policy=policy,
    )
    runtime_recorded = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        res_recorded.state.owner_checkpoint, kernel_limits=LIMITS
    )

    # Reception and interpretation were recorded
    assert len(runtime_recorded.experience_records) == 2
    rec_reception = runtime_recorded.experience_records[0]
    rec_interp = runtime_recorded.experience_records[1]

    assert rec_reception.source_kind is SourceKind.DIRECT
    assert rec_reception.epistemic_status is EpistemicStatus.OBSERVED
    assert rec_reception.evidence_refs == ("sense:world-1:4",)

    assert rec_interp.source_kind is SourceKind.COGNITIVE
    assert rec_interp.epistemic_status is EpistemicStatus.ASSOCIATED
    assert rec_interp.evidence_refs == ()

    # Checkpoint serialization round-trip
    chk_bytes = runtime_recorded.to_bytes()
    restored = CognitiveRuntimeCheckpoint.from_bytes(chk_bytes, kernel_limits=LIMITS)
    assert restored.experience_records == runtime_recorded.experience_records
