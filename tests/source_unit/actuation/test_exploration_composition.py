from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from organism.actuation import (
    ActuatorEvidenceContext,
    ActuatorSurface,
    CompetenceDevelopmentEngine,
    OrganismCheckpointContract,
    OrganismExplorationComposer,
    ORGANISM_TRANSITION_CHECKPOINT_SCHEMA,
    RestoredOrganismExplorationCheckpoint,
    RestoredOrganismTransitionCheckpoint,
)
from organism.cognition import (
    CognitiveRuntimeCheckpoint,
    CognitiveTransitionState,
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    advance_organism_from_sense_signals,
)
from organism.cognition.graph import CognitiveGraph, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.activation import SensoryNormalizer
from organism.cognition.metaplasticity import SafetyState
from organism.cognition.sense_input import (
    SCALAR_FIELD_CONTRACT_ID,
    SCALAR_FIELD_CONTRACT_REVISION,
    SCALAR_FIELD_QUANTITY_KIND,
    SCALAR_FIELD_UNIT,
)
from organism.cognition.types import NodeKind
from organism.identity import OrganismIdentity
from organism.checkpoint import CheckpointDependency, CheckpointDependencyMismatch
from organism.transition import (
    EmbodimentContextRef,
    OrganismLifecycle,
    OrganismTransitionState,
    WorldCausalCorrelation,
    transition_organism_lifecycle,
)
from organism.memory import EpistemicStatus, ExperienceRecord, SourceKind


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
        nodes=(
            PlasticNode("node.1", NodeKind.STATE),
            PlasticNode("sense.channel", NodeKind.SENSE),
        ),
        edges=(),
        kernel_limits=KernelLimits(),
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


def _candidate_with_command(
    composer: OrganismExplorationComposer,
    surface: ActuatorSurface,
    context: ActuatorEvidenceContext,
    *,
    dependencies: tuple[CheckpointDependency, ...] = (),
):
    for tick in range(1, 65):
        current_transition = _transition(tick=tick)
        if dependencies:
            current_transition = replace(
                current_transition,
                owner_checkpoint=replace(
                    current_transition.owner_checkpoint,
                    dependencies=dependencies,
                ),
            )
        candidate = composer.prepare(
            current_transition,
            actuator_context=context,
            surface=surface,
            target_causal_tick=tick + 1,
            physical_step_id=f"physical-step.{tick}",
            commit_id=f"commit.{tick}",
        )
        if candidate.motor_command is not None:
            return candidate
    pytest.fail("deterministic bounded exploration produced no command in 64 ticks")


def test_production_composer_stages_identity_bound_exploration_without_evidence_claims(tmp_path: Path) -> None:
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
            committed_cut=_CommittedCut("world.1", first.source_causal_tick, first.commit_id),
        )
    encoded = composer.checkpoint_for_candidate(
        first,
        transition_state=_transition(tick=first.source_causal_tick),
    )
    checkpoint_contract = OrganismCheckpointContract()
    schema = "biopeia-organism-exploration-checkpoint-v4"
    assert checkpoint_contract.validate_checkpoint(encoded, str(IDENTITY), schema) is True
    assert checkpoint_contract.validate_checkpoint(encoded, "other-organism", schema) is False
    assert checkpoint_contract.validate_checkpoint(encoded, str(IDENTITY), "unknown-schema") is False
    assert checkpoint_contract.validate_checkpoint(
        b'{"schema":"duplicate","schema":"duplicate"}', str(IDENTITY), schema
    ) is False
    transition_bytes = _transition(tick=first.source_causal_tick).to_bytes()
    assert checkpoint_contract.validate_checkpoint(
        transition_bytes, str(IDENTITY), ORGANISM_TRANSITION_CHECKPOINT_SCHEMA
    ) is True
    assert checkpoint_contract.validate_checkpoint(
        transition_bytes, "other-organism", ORGANISM_TRANSITION_CHECKPOINT_SCHEMA
    ) is False
    checkpoint_path = tmp_path / "organism-checkpoint.json"
    checkpoint_path.write_bytes(encoded)
    reference = SimpleNamespace(
        owner="organism",
        owner_identity=str(IDENTITY),
        schema_version=schema,
        locator=str(checkpoint_path),
        sha256=sha256(encoded).hexdigest(),
    )
    assert checkpoint_contract.verify_restoration(reference) is True
    assert checkpoint_contract.verify_restoration(
        SimpleNamespace(
            owner=reference.owner,
            owner_identity=reference.owner_identity,
            schema_version=reference.schema_version,
            locator=reference.locator,
            sha256="0" * 64,
        )
    ) is False
    restored = OrganismExplorationComposer.restore_from_committed_checkpoint(
        encoded,
        committed_cut=_CommittedCut("world.1", first.target_causal_tick, first.commit_id),
        expected_organism_id=str(IDENTITY),
        expected_physical_step_id=first.physical_step_id,
    )
    assert restored.cognitive_runtime.to_bytes() == (
        restored.transition_state.owner_checkpoint.private_state
    )
    with pytest.raises(ValueError, match="does not match committed World cut"):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            encoded,
            committed_cut=_CommittedCut("world.1", first.target_causal_tick, "commit.wrong"),
            expected_organism_id=str(IDENTITY),
            expected_physical_step_id=first.physical_step_id,
        )
    with pytest.raises(ValueError, match="expected owner or physical step"):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            encoded,
            committed_cut=_CommittedCut("world.1", first.target_causal_tick, first.commit_id),
            expected_organism_id="another-organism",
            expected_physical_step_id=first.physical_step_id,
        )
    with pytest.raises(ValueError, match="differs from the staged action source"):
        composer.checkpoint_for_candidate(
            first,
            transition_state=transition_organism_lifecycle(
                _transition(tick=first.source_causal_tick),
                OrganismLifecycle.DORMANT,
                transition_id="lifecycle.checkpoint-mismatch",
                request_ref="test.checkpoint-mismatch",
            ),
        )
    duplicated_schema = encoded.replace(
        b'{"action_domain":', b'{"schema":"biopeia-organism-exploration-checkpoint-v4","action_domain":', 1
    )
    with pytest.raises(ValueError, match="duplicate checkpoint field: schema"):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            duplicated_schema,
            committed_cut=_CommittedCut("world.1", first.target_causal_tick, first.commit_id),
            expected_organism_id=str(IDENTITY),
            expected_physical_step_id=first.physical_step_id,
        )
    outer = json.loads(encoded)
    transition_bytes = base64.b64decode(outer["transition_state_base64"])
    transition_json = transition_bytes.decode("utf-8")
    duplicate_transition_json = transition_json.replace(
        '"organism_step":1,', '"organism_step":1,"organism_step":1,', 1
    )
    assert duplicate_transition_json != transition_json
    outer["transition_state_base64"] = base64.b64encode(
        duplicate_transition_json.encode("utf-8")
    ).decode("ascii")
    with pytest.raises(ValueError, match="invalid embedded Organism transition checkpoint"):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            json.dumps(outer).encode("utf-8"),
            committed_cut=_CommittedCut("world.1", first.target_causal_tick, first.commit_id),
            expected_organism_id=str(IDENTITY),
            expected_physical_step_id=first.physical_step_id,
        )
    composer.adopt_after_world_commit(
        first,
        committed_cut=_CommittedCut("world.1", first.target_causal_tick, first.commit_id),
    )
    restored_composer = restored.action_composer
    assert restored_composer._generation == composer._generation
    assert restored_composer._engine.checkpoint() == composer._engine.checkpoint()
    assert restored_composer._domain.checkpoint() == composer._domain.checkpoint()

    # Equal serialized owner state is not sufficient evidence by itself: the
    # restored composition must consume that state identically on the next
    # accepted source transition. Compare both action output and the next
    # staged owner checkpoint under identical inputs.
    next_transition = _transition(tick=first.target_causal_tick)
    next_step_id = f"physical-step.{first.target_causal_tick}"
    next_commit_id = f"commit.{first.target_causal_tick}"
    next_candidate = composer.prepare(
        next_transition,
        actuator_context=context,
        surface=surface,
        target_causal_tick=first.target_causal_tick + 1,
        physical_step_id=next_step_id,
        commit_id=next_commit_id,
    )
    restored_candidate = restored_composer.prepare(
        next_transition,
        actuator_context=context,
        surface=surface,
        target_causal_tick=first.target_causal_tick + 1,
        physical_step_id=next_step_id,
        commit_id=next_commit_id,
    )
    assert restored_candidate.proposal == next_candidate.proposal
    assert restored_candidate.motor_command == next_candidate.motor_command
    assert restored_candidate.candidate_id == next_candidate.candidate_id
    assert restored_composer.checkpoint_for_candidate(
        restored_candidate, transition_state=next_transition
    ) == composer.checkpoint_for_candidate(
        next_candidate, transition_state=next_transition
    )
    with pytest.raises(ValueError, match="stale"):
        composer.adopt_after_world_commit(
            first,
            committed_cut=_CommittedCut("world.1", first.target_causal_tick, "commit.1"),
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
            target_causal_tick=2,
            physical_step_id="physical-step.1",
            commit_id="commit.1",
        )

    wrong_surface = ActuatorSurface.from_effector_ids(ACTUATORS, fingerprint_material="changed")
    with pytest.raises(ValueError, match="fingerprint"):
        composer.prepare(
            _transition(),
            actuator_context=context,
            surface=wrong_surface,
            target_causal_tick=2,
            physical_step_id="physical-step.1",
            commit_id="commit.1",
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
            target_causal_tick=2,
            physical_step_id="physical-step.1",
            commit_id="commit.1",
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
    assert result_a.source_causal_tick == result_b.source_causal_tick
    assert result_a.target_causal_tick == result_b.target_causal_tick
    assert result_a.motor_command == result_b.motor_command


def test_restored_committed_checkpoint_can_continue_at_next_world_cut() -> None:
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    composer = _composer(surface, context)
    candidate = _candidate_with_command(composer, surface, context)
    source_transition = _transition(tick=candidate.source_causal_tick)
    encoded = composer.checkpoint_for_candidate(
        candidate,
        transition_state=source_transition,
    )
    receipt = _CommittedCut(
        candidate.world_id,
        candidate.target_causal_tick,
        candidate.commit_id,
    )
    restored = OrganismExplorationComposer.restore_from_committed_checkpoint(
        encoded,
        committed_cut=receipt,
        expected_organism_id=str(IDENTITY),
        expected_physical_step_id=candidate.physical_step_id,
    )
    composer.adopt_after_world_commit(candidate, committed_cut=receipt)

    # The saved cognitive state represents the input consumed at the previous
    # source cut. It remains valid as the base for the next World-owned sample;
    # it must not be relabelled as if it already consumed that next sample.
    next_tick = candidate.target_causal_tick
    assert restored.transition_state.world_correlation == WorldCausalCorrelation(
        candidate.world_id, candidate.source_causal_tick
    )
    provenance = SenseProvenance(
        world_id=candidate.world_id,
        causal_tick=next_tick,
        physical_step_id=f"physical-step.{next_tick}",
        environment_id="environment.1",
        environment_revision="environment.r1",
        sample_frame_id="world.frame",
        sample_position_m=(0.0, 0.0, 0.0),
        body_id="body.1",
        embodiment_episode_id="episode.1",
        morphology_revision="morphology.r1",
    )
    signal = OpaqueSenseSignal(
        channel_id="channel.1",
        receptor_id="receptor.1",
        value=0.25,
        provenance=provenance,
        contract_id=SCALAR_FIELD_CONTRACT_ID,
        contract_revision=SCALAR_FIELD_CONTRACT_REVISION,
        quantity_kind=SCALAR_FIELD_QUANTITY_KIND,
        unit=SCALAR_FIELD_UNIT,
    )
    next_transition = advance_organism_from_sense_signals(
        restored.transition_state,
        (SenseChannelBinding("channel.1", "receptor.1", "sense.channel"),),
        (signal,),
        expected_provenance=provenance,
        kernel_limits=KernelLimits(),
    ).state
    uninterrupted_transition = advance_organism_from_sense_signals(
        source_transition,
        (SenseChannelBinding("channel.1", "receptor.1", "sense.channel"),),
        (signal,),
        expected_provenance=provenance,
        kernel_limits=KernelLimits(),
    ).state
    assert uninterrupted_transition == next_transition
    assert next_transition.organism_step == restored.transition_state.organism_step + 1
    assert next_transition.world_correlation == WorldCausalCorrelation(
        candidate.world_id, next_tick
    )

    next_candidate = restored.action_composer.prepare(
        next_transition,
        actuator_context=context,
        surface=surface,
        target_causal_tick=next_tick + 1,
        physical_step_id=f"physical-step.{next_tick}",
        commit_id=f"commit.{next_tick}",
    )
    uninterrupted_candidate = composer.prepare(
        uninterrupted_transition,
        actuator_context=context,
        surface=surface,
        target_causal_tick=next_tick + 1,
        physical_step_id=f"physical-step.{next_tick}",
        commit_id=f"commit.{next_tick}",
    )
    assert next_candidate.source_causal_tick == next_tick
    assert next_candidate.target_causal_tick == next_tick + 1
    assert next_candidate.proposal == uninterrupted_candidate.proposal
    assert next_candidate.motor_command == uninterrupted_candidate.motor_command
    assert restored.action_composer.checkpoint_for_candidate(
        next_candidate,
        transition_state=next_transition,
    ) == composer.checkpoint_for_candidate(
        uninterrupted_candidate,
        transition_state=uninterrupted_transition,
    )


def test_restore_verifies_embedded_checkpoint_dependency_closure() -> None:
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    dependency_bytes = b"external cognitive artifact"
    dependency = CheckpointDependency(
        locator="artifact/cognition-v1",
        sha256_hex=sha256(dependency_bytes).hexdigest(),
    )
    composer = _composer(surface, context)
    candidate = _candidate_with_command(
        composer,
        surface,
        context,
        dependencies=(dependency,),
    )
    source_transition = _transition(tick=candidate.source_causal_tick)
    dependent_transition = replace(
        source_transition,
        owner_checkpoint=replace(
            source_transition.owner_checkpoint,
            dependencies=(dependency,),
        ),
    )
    encoded = composer.checkpoint_for_candidate(
        candidate,
        transition_state=dependent_transition,
    )
    receipt = _CommittedCut("world.1", candidate.target_causal_tick, candidate.commit_id)
    restore_args = {
        "committed_cut": receipt,
        "expected_organism_id": str(IDENTITY),
        "expected_physical_step_id": candidate.physical_step_id,
    }
    with pytest.raises(ValueError, match="dependencies require a dependency reader"):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            encoded,
            **restore_args,
        )
    with pytest.raises(CheckpointDependencyMismatch):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            encoded,
            read_dependency=lambda locator: b"wrong bytes",
            **restore_args,
        )
    restored = OrganismExplorationComposer.restore_from_committed_checkpoint(
        encoded,
        read_dependency=lambda locator: dependency_bytes,
        **restore_args,
    )
    assert restored.transition_state == dependent_transition


def test_target_tick_must_match_world_physical_step_successor() -> None:
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    composer = _composer(surface, context)

    with pytest.raises(ValueError, match="immediately follow"):
        composer.prepare(
            _transition(tick=7),
            actuator_context=context,
            surface=surface,
            target_causal_tick=7,
            physical_step_id="physical-step.7",
            commit_id="commit.7",
        )


def test_owner_checkpoint_contract_validates_private_cognitive_payload() -> None:
    contract = OrganismCheckpointContract()
    valid_transition = _transition()
    assert contract.validate_checkpoint(
        valid_transition.to_bytes(),
        str(IDENTITY),
        ORGANISM_TRANSITION_CHECKPOINT_SCHEMA,
    ) is True

    # The generic envelope remains structurally valid and identity-matched,
    # but the owner-private payload is not a valid supported cognitive state.
    malformed_owner_checkpoint = replace(
        valid_transition.owner_checkpoint,
        private_state=b"not-a-cognitive-runtime-checkpoint",
    )
    malformed_transition = replace(
        valid_transition,
        owner_checkpoint=malformed_owner_checkpoint,
    )
    assert contract.validate_checkpoint(
        malformed_transition.to_bytes(),
        str(IDENTITY),
        ORGANISM_TRANSITION_CHECKPOINT_SCHEMA,
    ) is False


def test_exploration_restore_rejects_malformed_embedded_cognitive_payload() -> None:
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    composer = _composer(surface, context)
    candidate = _candidate_with_command(composer, surface, context)
    transition = _transition(tick=candidate.source_causal_tick)
    encoded = composer.checkpoint_for_candidate(
        candidate,
        transition_state=transition,
    )
    payload = json.loads(encoded)
    malformed_transition = replace(
        transition,
        owner_checkpoint=replace(
            transition.owner_checkpoint,
            private_state=b"not-a-cognitive-runtime-checkpoint",
        ),
    )
    payload["transition_state_base64"] = base64.b64encode(
        malformed_transition.to_bytes()
    ).decode("ascii")

    with pytest.raises(ValueError, match="malformed cognitive runtime checkpoint"):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            json.dumps(payload).encode("utf-8"),
            committed_cut=_CommittedCut(
                candidate.world_id,
                candidate.target_causal_tick,
                candidate.commit_id,
            ),
            expected_organism_id=str(IDENTITY),
            expected_physical_step_id=candidate.physical_step_id,
        )


def test_restored_exploration_result_rejects_same_identity_with_different_payload() -> None:
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    composer = _composer(surface, context)
    candidate = _candidate_with_command(composer, surface, context)
    encoded = composer.checkpoint_for_candidate(
        candidate,
        transition_state=_transition(tick=candidate.source_causal_tick),
    )
    restored = OrganismExplorationComposer.restore_from_committed_checkpoint(
        encoded,
        committed_cut=_CommittedCut(
            candidate.world_id,
            candidate.target_causal_tick,
            candidate.commit_id,
        ),
        expected_organism_id=str(IDENTITY),
        expected_physical_step_id=candidate.physical_step_id,
    )
    inconsistent_runtime = replace(
        restored.cognitive_runtime,
        state=replace(restored.cognitive_runtime.state, last_causal_tick=99),
    )

    with pytest.raises(ValueError, match="does not match transition checkpoint"):
        RestoredOrganismExplorationCheckpoint(
            transition_state=restored.transition_state,
            cognitive_runtime=inconsistent_runtime,
            action_composer=restored.action_composer,
        )


def test_owner_checkpoint_contract_rejects_oversized_bytes_before_decoding() -> None:
    encoded = _transition().to_bytes()
    contract = OrganismCheckpointContract()
    oversized = encoded + b" " * (
        KernelLimits().max_owner_checkpoint_bytes - len(encoded) + 1
    )

    assert contract.validate_checkpoint(
        encoded, str(IDENTITY), ORGANISM_TRANSITION_CHECKPOINT_SCHEMA
    ) is True
    assert contract.validate_checkpoint(
        oversized, str(IDENTITY), ORGANISM_TRANSITION_CHECKPOINT_SCHEMA
    ) is False


def test_owner_checkpoint_contract_rejects_private_identity_mismatch() -> None:
    contract = OrganismCheckpointContract()
    transition = _transition()
    different_identity = OrganismIdentity.create(
        UUID("d6e5d7f6-e8a2-451e-a9c4-62c27be2c47d")
    )
    private_checkpoint = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState.initial(different_identity),
        graph=CognitiveGraph(
            nodes=(PlasticNode("node.1", NodeKind.STATE),),
            edges=(),
            kernel_limits=KernelLimits(),
        ),
    ).to_organism_checkpoint()
    mismatched_transition = replace(
        transition,
        owner_checkpoint=replace(
            private_checkpoint,
            identity=transition.identity,
        ),
    )
    assert contract.validate_checkpoint(
        mismatched_transition.to_bytes(),
        str(IDENTITY),
        ORGANISM_TRANSITION_CHECKPOINT_SCHEMA,
    ) is False


def test_owner_contract_restores_supported_transition_profile_before_returning() -> None:
    contract = OrganismCheckpointContract()
    transition_state = _transition(tick=4)
    safety = SafetyState()
    safety.record_failure()
    normalizer = SensoryNormalizer()
    for value in (0.1, 0.4, 0.7, 0.8):
        normalizer.normalize(value)
    source_runtime = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState(
            identity=IDENTITY,
            last_causal_tick=4,
            previous_activations=(("sense.channel", 0.625),),
        ),
        graph=CognitiveGraph(
            nodes=(
                PlasticNode("node.1", NodeKind.STATE),
                PlasticNode("sense.channel", NodeKind.SENSE),
            ),
            edges=(),
            kernel_limits=KernelLimits(),
        ),
        experience_records=(
            ExperienceRecord(
                record_id="experience.1",
                organism_id=str(IDENTITY),
                tick_class=4,
                context_tokens=("opaque.context",),
                action_token="opaque.action",
                outcome_tokens=("opaque.outcome",),
                epistemic_status=EpistemicStatus.OBSERVED,
                evidence_refs=("causal-record.4",),
                confidence_class=3,
                source_kind=SourceKind.DIRECT,
            ),
        ),
        safety_state=safety,
        sensory_normalizers={"sense.channel": normalizer},
    )
    source = replace(
        transition_state,
        owner_checkpoint=source_runtime.to_organism_checkpoint(),
    )

    restored = contract.restore_transition_checkpoint(
        source.to_bytes(),
        str(IDENTITY),
    )

    assert isinstance(restored, RestoredOrganismTransitionCheckpoint)
    assert restored.transition_state == source
    assert restored.cognitive_runtime.state == source_runtime.state
    assert restored.cognitive_runtime.experience_records == source_runtime.experience_records
    assert restored.cognitive_runtime.safety_state == safety
    restored_normalizer = restored.cognitive_runtime.sensory_normalizers["sense.channel"]
    assert restored_normalizer.mean == pytest.approx(normalizer.mean)
    assert restored_normalizer.variance == pytest.approx(normalizer.variance)
    assert restored_normalizer.count == normalizer.count
    assert restored.cognitive_runtime.graph.nodes == source_runtime.graph.nodes
    assert restored.cognitive_runtime.graph.edges == source_runtime.graph.edges
    assert restored.cognitive_runtime.graph.kernel_limits == source_runtime.graph.kernel_limits
    # The complete serialized profile represented by this fixture remains
    # byte-stable through owner restore and re-encoding.
    assert (
        restored.cognitive_runtime.to_organism_checkpoint().to_bytes()
        == source.owner_checkpoint.to_bytes()
    )

    provenance = SenseProvenance(
        world_id="world.1",
        causal_tick=5,
        physical_step_id="physical-step.5",
        environment_id="environment.1",
        environment_revision="environment.r1",
        sample_frame_id="world.frame",
        sample_position_m=(0.0, 0.0, 0.0),
        body_id="body.1",
        embodiment_episode_id="episode.1",
        morphology_revision="morphology.r1",
    )
    signal = OpaqueSenseSignal(
        channel_id="channel.1",
        receptor_id="receptor.1",
        value=0.25,
        provenance=provenance,
        contract_id=SCALAR_FIELD_CONTRACT_ID,
        contract_revision=SCALAR_FIELD_CONTRACT_REVISION,
        quantity_kind=SCALAR_FIELD_QUANTITY_KIND,
        unit=SCALAR_FIELD_UNIT,
    )
    continued = advance_organism_from_sense_signals(
        restored.transition_state,
        (SenseChannelBinding("channel.1", "receptor.1", "sense.channel"),),
        (signal,),
        expected_provenance=provenance,
        kernel_limits=KernelLimits(),
    )
    assert continued.state.organism_step == source.organism_step + 1
    assert continued.state.identity == source.identity


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda state: replace(state, owner_checkpoint=replace(state.owner_checkpoint, private_state=b"invalid")), ValueError),
        (
            lambda state: replace(
                state,
                owner_checkpoint=replace(
                    state.owner_checkpoint,
                    dependencies=(CheckpointDependency("external", "0" * 64),),
                ),
            ),
            ValueError,
        ),
    ],
)
def test_owner_contract_does_not_return_partially_restored_transition(
    mutate, expected_error
) -> None:
    contract = OrganismCheckpointContract()
    source = mutate(_transition())

    with pytest.raises(expected_error):
        contract.restore_transition_checkpoint(source.to_bytes(), str(IDENTITY))


def test_owner_contract_rejects_wrong_identity_and_oversized_transition_before_restore() -> None:
    source = _transition().to_bytes()
    contract = OrganismCheckpointContract()

    with pytest.raises(ValueError, match="expected owner"):
        contract.restore_transition_checkpoint(source, "organism.other")
    oversized = source + b" " * (KernelLimits().max_owner_checkpoint_bytes - len(source) + 1)
    with pytest.raises(ValueError, match="byte limit"):
        contract.restore_transition_checkpoint(oversized, str(IDENTITY))


def test_restored_transition_result_rejects_inconsistent_component_identity() -> None:
    contract = OrganismCheckpointContract()
    restored = contract.restore_transition_checkpoint(
        _transition().to_bytes(), str(IDENTITY)
    )
    other_identity = OrganismIdentity.create(
        UUID("d6e5d7f6-e8a2-451e-a9c4-62c27be2c47d")
    )
    inconsistent_runtime = replace(
        restored.cognitive_runtime,
        state=CognitiveTransitionState.initial(other_identity),
    )

    with pytest.raises(ValueError, match="identities differ"):
        RestoredOrganismTransitionCheckpoint(
            transition_state=restored.transition_state,
            cognitive_runtime=inconsistent_runtime,
        )


def test_restored_transition_result_rejects_same_identity_with_different_payload() -> None:
    contract = OrganismCheckpointContract()
    restored = contract.restore_transition_checkpoint(
        _transition().to_bytes(), str(IDENTITY)
    )
    inconsistent_runtime = replace(
        restored.cognitive_runtime,
        state=replace(restored.cognitive_runtime.state, last_causal_tick=99),
    )

    with pytest.raises(ValueError, match="does not match transition checkpoint"):
        RestoredOrganismTransitionCheckpoint(
            transition_state=restored.transition_state,
            cognitive_runtime=inconsistent_runtime,
        )
