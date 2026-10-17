from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import replace
import subprocess
import sys
from uuid import UUID

import pytest

from organism import (
    CheckpointDependency,
    CheckpointDependencyMismatch,
    InvalidLongitudinalCheckpoint,
    OrganismCheckpoint,
    OrganismIdentity,
    OrganismLongitudinalCheckpoint,
)
from organism.checkpoint import REQUIRED_LONGITUDINAL_COMPONENTS
from organism.checkpoint import (
    LongitudinalComponentActivity,
    LongitudinalComponentClassification,
)


def _classifications():
    return {
        name: LongitudinalComponentClassification(
            LongitudinalComponentActivity.ACTIVE, f"test-profile:{name}"
        )
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }


def _checkpoint() -> OrganismLongitudinalCheckpoint:
    return OrganismLongitudinalCheckpoint(
        identity=OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")),
        components={name: f"{name}:payload".encode() for name in REQUIRED_LONGITUDINAL_COMPONENTS},
        classifications=_classifications(),
    )


def test_longitudinal_checkpoint_round_trips_exact_component_bytes() -> None:
    checkpoint = _checkpoint()
    restored = OrganismLongitudinalCheckpoint.from_bytes(checkpoint.to_bytes())

    assert restored.identity == checkpoint.identity
    assert dict(restored.components) == dict(checkpoint.components)
    assert dict(restored.classifications) == dict(checkpoint.classifications)
    assert restored.to_bytes() == checkpoint.to_bytes()
    assert OrganismLongitudinalCheckpoint.from_organism_checkpoint(
        checkpoint.to_organism_checkpoint()
    ).to_bytes() == checkpoint.to_bytes()


def test_longitudinal_checkpoint_rejects_missing_or_unknown_components() -> None:
    components = dict(_checkpoint().components)
    components.pop("agency")
    with pytest.raises(InvalidLongitudinalCheckpoint, match="payload mismatch"):
        OrganismLongitudinalCheckpoint(
            OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")), components,
            _classifications(),
        )
    components["agency"] = b"agency"
    components["unregistered"] = b"must not be silently ignored"
    with pytest.raises(InvalidLongitudinalCheckpoint, match="payload mismatch"):
        OrganismLongitudinalCheckpoint(
            OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")), components,
            _classifications(),
        )


def test_capture_uses_a_complete_identity_bound_encoder_registry() -> None:
    identity = OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041"))
    states = {name: name for name in REQUIRED_LONGITUDINAL_COMPONENTS}
    encoders = {
        name: lambda value, owner: f"{owner}:{value}".encode()
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    checkpoint = OrganismLongitudinalCheckpoint.capture_components(
        identity, states, encoders, classifications=_classifications()
    )

    assert all(str(identity).encode() in payload for payload in checkpoint.components.values())
    with pytest.raises(InvalidLongitudinalCheckpoint, match="encoder registry"):
        OrganismLongitudinalCheckpoint.capture_components(
            identity,
            states,
            {key: value for key, value in encoders.items() if key != "agency"},
            classifications=_classifications(),
        )

def test_longitudinal_checkpoint_rejects_tampered_component_digest() -> None:
    document = json.loads(_checkpoint().to_bytes())
    document["components"]["cognitive_graph"]["payload_base64"] = "Y29ycnVwdGVk"

    with pytest.raises(InvalidLongitudinalCheckpoint, match="digest mismatch"):
        OrganismLongitudinalCheckpoint.from_bytes(
            json.dumps(document, separators=(",", ":")).encode()
        )


def test_longitudinal_checkpoint_rejects_duplicate_json_keys() -> None:
    payload = _checkpoint().to_bytes()
    duplicate = payload[:-1] + b',"schema_version":2}'
    with pytest.raises(InvalidLongitudinalCheckpoint, match="malformed"):
        OrganismLongitudinalCheckpoint.from_bytes(duplicate)


def test_restore_requires_complete_decoders_and_dependency_closure() -> None:
    checkpoint = _checkpoint()
    decoders = {
        name: lambda payload, identity: payload.decode()
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    with pytest.raises(InvalidLongitudinalCheckpoint, match="decoder registry"):
        checkpoint.restore_components(
            {key: value for key, value in decoders.items() if key != "agency"},
            read_dependency=lambda locator: b"artifact",
        )

    dependency = b"model weights"
    checkpoint = OrganismLongitudinalCheckpoint(
        identity=checkpoint.identity,
        components=checkpoint.components,
        classifications=checkpoint.classifications,
        dependencies=(CheckpointDependency("model://predictor", sha256(dependency).hexdigest()),),
    )

    def fail_decode(payload: bytes, identity: OrganismIdentity) -> object:
        raise ValueError("invalid subsystem payload")

    with pytest.raises(InvalidLongitudinalCheckpoint, match="failed to restore"):
        checkpoint.restore_components(
            {**decoders, "agency": fail_decode},
            read_dependency=lambda locator: dependency,
        )
    with pytest.raises(CheckpointDependencyMismatch, match="model://predictor"):
        checkpoint.restore_components(decoders, read_dependency=lambda locator: b"wrong")


def test_longitudinal_checkpoint_captures_and_restores_action_domain() -> None:
    from organism.actuation.domain import ActionDomain

    identity = OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041"))
    domain = ActionDomain()
    states = {name: name for name in REQUIRED_LONGITUDINAL_COMPONENTS}
    states["action_commitments"] = domain

    encoders = {
        name: (lambda value, owner: f"{owner}:{value}".encode())
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    encoders["action_commitments"] = lambda value, owner: value.to_bytes()

    checkpoint = OrganismLongitudinalCheckpoint.capture_components(
        identity, states, encoders, classifications=_classifications()
    )
    assert "action_commitments" in checkpoint.components

    decoders = {
        name: (lambda payload, owner: payload.decode())
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    decoders["action_commitments"] = lambda payload, owner: ActionDomain.from_bytes(payload)

    restored_components = checkpoint.restore_components(
        decoders, read_dependency=lambda locator: b""
    )
    restored_domain = restored_components["action_commitments"]
    assert isinstance(restored_domain, ActionDomain)
    assert restored_domain.checkpoint() == domain.checkpoint()


def test_typed_vital_composition_restores_in_fresh_process_with_nondefault_configuration(tmp_path) -> None:
    """M2 restores owner state, not merely a structurally valid aggregate."""
    from organism.actuation.binding import (CompetenceExecutionBindingRegistry, BindingStatus, StalenessReason)
    from organism.actuation.competence import (
        CompetenceEvidence, CompetenceGate, CompetenceLibrary, MotorCompetence,
    )
    from organism.actuation.domain import ActionDomain
    from organism.actuation.sensorimotor import CompetenceDevelopmentEngine
    from organism.actuation.evidence import CausalEvidenceLedger, SensorimotorTransition
    from organism.actuation.model import CompetenceEffectModel, ControllabilityModel, AgencyModel
    from organism.actuation.context import ActuatorEvidenceContext
    from organism.agency.intention import IntentStatus
    from organism.agency.policy import ProspectivePolicy
    from organism.agency.prospective import ProspectiveAgency
    from organism.agency.value import OutcomeValueLedger
    from organism.cognition.activation import SensoryNormalizer
    from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
    from organism.cognition.composition import EdgeLearningInstruction, compose_cognition_transition
    from organism.cognition.types import EdgeKind, NodeKind
    from organism.cognition.limits import KernelLimits
    from organism.cognition.metaplasticity import SafetyState
    from organism.cognition.portable_checkpoint import CognitiveRuntimeCheckpoint
    from organism.cognition.transition import CognitiveTransitionState
    from organism.genetics import EpigeneticMark, GeneExpressionState, GermlineState, create_standard_genome
    from organism.longitudinal import OrganismVitalRestoreContext, OrganismVitalRuntime, OrganismVitalState
    from organism.memory import ExperienceLedger
    from organism.memory import EpistemicStatus, ExperienceRecord, SourceKind
    from organism.transition import EmbodimentContextRef, OrganismLifecycle, OrganismTransitionState, WorldCausalCorrelation

    identity = OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041"))
    limits = KernelLimits(max_nodes=512)
    transition = OrganismTransitionState(
        owner_checkpoint=OrganismCheckpoint(identity, b"temporary", ()),
        organism_step=23,
        embodiment=EmbodimentContextRef("body.1", "episode.1", "morphology.1"),
        world_correlation=WorldCausalCorrelation("world.1", 123),
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    genome = create_standard_genome("vital-state-test")
    germline = GermlineState.from_genome(
        genome, inherited_marks=(EpigeneticMark("plasticity.learning_rate.baseline", delta=0.01, strength=0.5),),
        acquired_capture_enabled=True,
    )
    expression = GeneExpressionState.from_genome(genome, germline=germline)
    normalizer = SensoryNormalizer()
    normalizer.normalize(0.25)
    normalizer.normalize(0.75)
    initial_graph = CognitiveGraph(
        nodes=(PlasticNode("sense", NodeKind.SENSE), PlasticNode("memory", NodeKind.STATE)),
        edges=(PlasticEdge("sense", "memory", EdgeKind.EXCITATORY, 0.5, 0.5, 1),),
        kernel_limits=limits,
    )
    first = compose_cognition_transition(CognitiveTransitionState.initial(identity), initial_graph,
                                         {"sense": 0.8}, causal_tick=0)
    learned = compose_cognition_transition(
        first.state, first.graph, {"sense": 0.6}, causal_tick=1,
        learning=(EdgeLearningInstruction("sense", "memory", kind=EdgeKind.EXCITATORY,
            learning_rate=0.1, modulation=1.0, eligibility_decay=0.5, eligible=True),),
    )
    learned_graph = learned.graph
    assert learned_graph.edges[0].weight != initial_graph.edges[0].weight
    cognition = CognitiveRuntimeCheckpoint(
        state=learned.state, graph=learned_graph,
        experience_records=(ExperienceRecord(
            record_id="experience.vital.1", organism_id=str(identity), tick_class=12,
            context_tokens=("sense",), action_token="action.1", outcome_tokens=("outcome.1",),
            epistemic_status=EpistemicStatus.OBSERVED, evidence_refs=("record.12",),
            confidence_class=6, source_kind=SourceKind.DIRECT,
        ),),
        safety_state=SafetyState(), sensory_normalizers={"sense": normalizer},
        genome=genome, germline_state=germline, expression_state=expression,
    )
    experience = ExperienceLedger(str(identity), max_records=64)
    policy = ProspectivePolicy(organism_id=str(identity), min_model_confidence=0.73,
                               min_value_samples=5, decision_margin=0.03)
    agency = ProspectiveAgency(organism_id=str(identity), outcome_value_ledger=OutcomeValueLedger(),
                               policy=policy, query_budget=5)
    actuator_ids = tuple(f"actuator.{i}" for i in range(4))
    from organism.actuation.surface import ActuatorSurface, derive_actuator_constitution
    surface = ActuatorSurface.from_effector_ids(actuator_ids)
    actuator_context = ActuatorEvidenceContext("body.1", "episode.1", "morphology.1", surface.contract_fingerprint)
    engine = CompetenceDevelopmentEngine(
        actuator_ids, organism_id=str(identity),
        embodiment_fingerprint=actuator_context.fingerprint, max_concurrent=2,
    )
    sequence = (
        {"actuator.0": 0.7, "actuator.1": 0.3}, {"actuator.0": 0.5, "actuator.2": 0.6},
        {"actuator.1": 0.6, "actuator.3": 0.4}, {"actuator.0": 0.3, "actuator.2": 0.7, "actuator.3": 0.2},
    )
    body = {"sense.a": 0.0, "sense.b": 0.0}
    tick = 0
    for episode in range(2):
        for step, vector in enumerate(sequence):
            engine.observe(tick=tick, body_state=body, motor_vector=vector, discovery_eligible=True)
            drive = sum(vector.values())
            body = {"sense.a": body["sense.a"] + drive * 0.01 * ((step + 1) / 4),
                    "sense.b": body["sense.b"] - drive * 0.006 * ((step + 1) / 4)}
            tick += 1
        engine.observe(tick=tick, body_state=body, motor_vector={}, discovery_eligible=False)
        tick += 1
        if episode == 0:
            while tick % 8:
                engine.observe(tick=tick, body_state=body, motor_vector={}, discovery_eligible=False)
                tick += 1
    assert engine.primitives and engine.snapshot().competence_chunks > 0
    engine.gate = CompetenceGate(min_support=3, min_consistency=0.8)
    from organism.actuation.context import ActuatorEvidenceContext
    bindings = CompetenceExecutionBindingRegistry(organism_id=str(identity))
    bindings.bind_from_evidence(
        competence_id="competence.pending", surface_fingerprint="surface.s",
        applicability_context=ActuatorEvidenceContext("body.1", "episode.1", "morphology.1", "surface.s"), effect_id="effect.e",
        evidence_refs=("evidence.1",), reliability=0.9, controllability=0.9, tick=1,
    )
    bindings.set_status("competence.pending", BindingStatus.STALE,
        StalenessReason.SURFACE_NOT_CURRENT, tick=2, cause_refs=("surface-revision-2",))
    assert bindings.checkpoint()["pending_transitions"]
    # These are additional live Organism-owned competence/evidence/model
    # owners; their native codecs preserve bounded capacity and configuration.
    competence_library = CompetenceLibrary(max_competences=17)
    competence_library.add(MotorCompetence(
        competence_id="competence.walk", controller_id="controller.walk",
        effect_id="effect.walk",
        evidence=CompetenceEvidence(
            controller_seed_ref="seed.walk", effect_evidence_refs=("evidence.motor.1",),
            context_evidence_refs=("context.walk",),
            controllability_evidence_refs=("evidence.control.1",), support=3,
            reproducibility=0.9, controllability=0.8, directional_consistency=0.85,
            gate=engine.gate,
        ),
    ))
    causal_evidence = CausalEvidenceLedger(capacity=31, passive_capacity=9)
    causal_evidence.observe(SensorimotorTransition(
        transition_id="transition.motor.1", tick_start=8, tick_end=9,
        context_ref="context.walk", commitment_id="commitment.walk",
        controller_id="controller.walk", competence_id="competence.walk",
        attempt_id="attempt.walk.1", intervention_signature_id="signature.walk",
        state_before_ref="state.before", motor_command_ref="motor.command",
        actuation_ref="actuation.1", prediction_ref="prediction.1",
        state_after_ref="state.after", observed_effect_id="effect.walk",
    ))
    causal_evidence.observe_passive_window(
        tick_start=9, tick_end=10, context_ref="context.walk",
        prior_state_ref="state.after", resulting_state_ref="state.passive.after",
        effect_id="effect.walk",
    )
    effect_model = CompetenceEffectModel(max_contexts=23)
    effect_model.rebuild(causal_evidence)
    controllability_model = ControllabilityModel()
    controllability_model.rebuild(causal_evidence)
    agency_model = AgencyModel()
    agency_model.rebuild(causal_evidence)
    from organism.agency.executive_outcome import ExecutiveOutcomeLedger
    actions = ActionDomain(organism_id=str(identity), outcome_ledger=ExecutiveOutcomeLedger())
    from organism.agency.affordance import ActionAffordance
    from organism.agency.prospective import ProspectiveDecision
    from organism.actuation.action import (ActionEvaluation, ActionJustification, ActionProposal, ActionSource)
    from organism.actuation.surface import derive_actuator_constitution
    from organism.agency.executive_outcome import ExecutiveOutcomeSample, OutcomeClass
    affordance = ActionAffordance(
        affordance_id="affordance.1", competence_id="competence.walk", anticipated_effect_id="effect.walk",
        context_ref="context.1", embodiment_id="body.1", prediction_confidence=0.5,
        controllability=0.4, executability_confidence=1.0, prediction_ref="prediction.1",
        evidence_refs=("evidence.1",),
    )
    decision = ProspectiveDecision(
        competence_id="competence.walk", anticipated_effect_id="effect.walk", prediction_ref="prediction.1",
        confidence=0.5, epistemic_relevance=0.4, homeostatic_relevance=0.2,
        origin_refs=("source.1",), supporting_affordance_id="affordance.1",
    )
    actions.admit(decision, supporting_affordance=affordance, intent_id="intent.closed",
                  context_ref="context.1", embodiment_id="body.1", tick=4)
    proposal = ActionProposal("proposal.1", ActionSource.COMPETENCE, "effect.walk", "competence.walk",
        ActionJustification(effect_target_id="effect.walk", competence_id="competence.walk"),
        ActionEvaluation(effect_confidence=0.5, uncertainty=0.2), intent_id="intent.closed")
    action_surface = derive_actuator_constitution(("effector.left",))
    selected = actions.select_and_commit((proposal,), tick=4, commitment_id="commitment.closed",
        controller_id="controller.1", surface=action_surface, embodiment_id="body.1",
        interruptibility=1.0, minimum_duration=0, maximum_duration=None)
    actions.close(selected.commitment.commitment_id, status=IntentStatus.SATISFIED,
        tick=5, reason="anticipated_effect_observed", outcome=ExecutiveOutcomeSample(tick=5, outcome_class=OutcomeClass.POSITIVE,
        reason="anticipated_effect_observed", effect_similarity=1.0, prediction_mismatch=0.0, progress_before_failure=None))
    active_affordance = replace(affordance, affordance_id="affordance.2", context_ref="context.2")
    active_decision = replace(decision, supporting_affordance_id="affordance.2")
    actions.admit(active_decision, supporting_affordance=active_affordance, intent_id="intent.active",
                  context_ref="context.2", embodiment_id="body.1", tick=6)
    active_proposal = replace(proposal, proposal_id="proposal.2", intent_id="intent.active")
    selected_active = actions.select_and_commit((active_proposal,), tick=6, commitment_id="commitment.active",
        controller_id="controller.2", surface=action_surface, embodiment_id="body.1",
        interruptibility=1.0, minimum_duration=0, maximum_duration=None)
    assert selected_active.commitment is actions.active_commitment
    assert actions._outcome_ledger.get(("competence.walk", "effect.walk")).satisfactions == 1
    source_transition = replace(
        transition, owner_checkpoint=OrganismCheckpoint(identity, cognition.to_bytes())
    )
    vital = OrganismVitalState(
        source_transition, cognition, experience, agency, engine, actions, bindings,
        competence_library, causal_evidence, effect_model, controllability_model, agency_model,
    )
    captured = vital.capture_transition()

    # A payload that declares the full-vital schema must never be reinterpreted
    # as the older cognitive-only format when its closure is malformed.
    malformed_document = json.loads(captured.owner_checkpoint.private_state)
    del malformed_document["components"]["experience_memory"]
    malformed_owner = OrganismCheckpoint(
        identity,
        json.dumps(malformed_document, sort_keys=True, separators=(",", ":")).encode(),
    )
    malformed_transition = replace(captured, owner_checkpoint=malformed_owner)
    from organism.actuation.bounded_action import (
        ORGANISM_TRANSITION_CHECKPOINT_SCHEMA, OrganismCheckpointContract,
    )
    contract = OrganismCheckpointContract(vital_restore_context=_restore_context())
    assert not contract.validate_checkpoint(
        malformed_transition.to_bytes(), str(identity), ORGANISM_TRANSITION_CHECKPOINT_SCHEMA,
    )
    with pytest.raises(InvalidLongitudinalCheckpoint):
        contract.restore_transition_checkpoint(
            malformed_transition.to_bytes(), str(identity),
        )

    # Exercise the owner-held production candidate seam. The runtime captures
    # all ten required active components and rejects caller-side drift. Keep
    # the committed active-intention fixture above for restore comparison;
    # the exploration composer correctly refuses to replace that commitment.
    candidate_actions = ActionDomain.from_bytes(actions.to_bytes())
    candidate_actions.close(
        "commitment.active", status=IntentStatus.INTERRUPTED, tick=7,
        reason="candidate-seam-test-boundary",
    )
    candidate_vital = replace(vital, actions=candidate_actions)
    from organism.actuation.bounded_action import OrganismExplorationComposer
    from organism.actuation.context import ActuatorEvidenceContext
    composer = OrganismExplorationComposer(
        organism_id=str(identity),
        engine=engine,
        action_domain=candidate_actions,
    )
    candidate = composer.prepare(
        source_transition, actuator_context=actuator_context, surface=surface,
        target_causal_tick=124, physical_step_id="physical-step.124", commit_id="commit.124",
    )
    vital_runtime = OrganismVitalRuntime(candidate_vital)
    # Production callers need not rebuild/pass the complete owner composition:
    # the runtime parses the cognition emitted by the transition and holds all
    # other accepted owners itself.
    candidate_bytes = vital_runtime.checkpoint_for_candidate(
        composer, candidate, transition_state=source_transition,
    )
    assert vital_runtime.checkpoint_for_candidate(
        composer, candidate, transition_state=source_transition,
        candidate_vital_state=candidate_vital,
    ) == candidate_bytes
    candidate_checkpoint = OrganismTransitionState.from_bytes(candidate_bytes)
    candidate_aggregate = OrganismLongitudinalCheckpoint.from_organism_checkpoint(
        candidate_checkpoint.owner_checkpoint
    )
    assert candidate_checkpoint.world_correlation == source_transition.world_correlation
    assert set(candidate_aggregate.components) == REQUIRED_LONGITUDINAL_COMPONENTS
    assert all(
        classification.activity is LongitudinalComponentActivity.ACTIVE
        for classification in candidate_aggregate.classifications.values()
    )
    accepted_candidate_source = OrganismLongitudinalCheckpoint.from_organism_checkpoint(
        candidate_vital.capture()
    )
    changed_components = {
        name for name in REQUIRED_LONGITUDINAL_COMPONENTS
        if candidate_aggregate.components[name] != accepted_candidate_source.components[name]
    }
    assert changed_components == {"competencies", "action_commitments"}
    candidate_engine = json.loads(candidate_aggregate.components["competencies"])
    assert candidate_engine["engine"]["embodiment_fingerprint"] == actuator_context.fingerprint
    assert candidate_engine["engine"]["primitives"] == engine.checkpoint()["primitives"]
    with pytest.raises(InvalidLongitudinalCheckpoint, match="another vital-state candidate"):
        next_candidate = replace(candidate, candidate_id="candidate.other")
        vital_runtime.checkpoint_for_candidate(
            composer, next_candidate, transition_state=source_transition,
            candidate_vital_state=candidate_vital,
        )
    # A candidate does not become the runtime's accepted baseline until the
    # composer has adopted it and the caller supplies the matching cut receipt.
    from types import SimpleNamespace
    cut_receipt = SimpleNamespace(
        world_id=candidate.world_id,
        causal_tick=candidate.target_causal_tick,
        commit_id=candidate.commit_id,
    )
    with pytest.raises(InvalidLongitudinalCheckpoint, match="has not adopted"):
        vital_runtime.adopt_after_world_commit(composer, candidate, committed_cut=cut_receipt)
    composer.adopt_after_world_commit(candidate, committed_cut=cut_receipt)
    vital_runtime.adopt_after_world_commit(composer, candidate, committed_cut=cut_receipt)
    assert vital_runtime.accepted_source_checkpoint.private_state == candidate_checkpoint.owner_checkpoint.private_state
    assert vital_runtime._pending_candidate_id is None
    experience.append(ExperienceRecord(
        record_id="experience.after-capture", organism_id=str(identity), tick_class=13,
        context_tokens=("sense",), action_token="action.late", outcome_tokens=(),
        epistemic_status=EpistemicStatus.OBSERVED, evidence_refs=("record.late",),
        confidence_class=5, source_kind=SourceKind.DIRECT,
    ))
    with pytest.raises(InvalidLongitudinalCheckpoint, match="changed outside a committed restore boundary"):
        vital_runtime.checkpoint_for_candidate(
            composer, candidate, transition_state=source_transition,
            candidate_vital_state=candidate_vital,
        )

    payload_path = tmp_path / "checkpoint.json"
    # Restore the staged candidate in a fresh process, not the accepted
    # baseline. This proves that the changed competence/action state survives
    # the same serialization boundary that World would consume.
    payload_path.write_bytes(candidate_bytes)
    aggregate = candidate_aggregate
    # The owner envelope alone cannot prove ownership of nested component
    # state. Both independently-restorable payloads carry and verify it.
    from organism.longitudinal import restore_vital_state
    for component, path in (("action_commitments", ()), ("competencies", ("bindings",))):
        for owner_id in ("00000000-0000-0000-0000-000000000099", None):
            components = dict(aggregate.components)
            component_payload = json.loads(components[component])
            target = component_payload
            for key in path:
                target = target[key]
            if owner_id is None:
                target.pop("organism_id")
            else:
                target["organism_id"] = owner_id
            components[component] = json.dumps(component_payload, sort_keys=True, separators=(",", ":")).encode()
            forged = OrganismLongitudinalCheckpoint(
                aggregate.identity, components, aggregate.classifications, aggregate.dependencies,
            )
            forged_owner = OrganismCheckpoint(aggregate.identity, forged.to_bytes())
            with pytest.raises(InvalidLongitudinalCheckpoint, match="failed to restore"):
                restore_vital_state(forged_owner, _restore_context())
    expected_max_concurrent = json.loads(aggregate.components["competencies"])["engine"]["max_concurrent"]
    expected_component_digests = {
        name: sha256(payload).hexdigest() for name, payload in aggregate.components.items()
    }
    digest_path = tmp_path / "component-digests.json"
    digest_path.write_text(json.dumps(expected_component_digests, sort_keys=True))

    # A new interpreter process uses only the serialized checkpoint and the
    # verified, non-checkpoint runtime configuration below.
    script = r"""
import sys
import json
from hashlib import sha256
from organism import OrganismTransitionState
from organism.cognition.limits import KernelLimits
from organism.actuation.competence import CompetenceGate
from organism.agency.policy import ProspectivePolicy
from organism import OrganismLongitudinalCheckpoint
from organism.longitudinal import (
    OrganismVitalRestoreContext, OrganismVitalRuntime, OrganismVitalState,
    restore_vital_state,
)
transition = OrganismTransitionState.from_bytes(open(sys.argv[1], 'rb').read())
identity = str(transition.identity)
policy = ProspectivePolicy(organism_id=identity, min_model_confidence=0.73, min_value_samples=5, decision_margin=0.03)
from organism.actuation.context import ActuatorEvidenceContext
from organism.actuation.surface import ActuatorSurface
actuator_ids = tuple(f'actuator.{i}' for i in range(4))
surface = ActuatorSurface.from_effector_ids(actuator_ids)
actuator_context = ActuatorEvidenceContext('body.1', 'episode.1', 'morphology.1', surface.contract_fingerprint)
context = OrganismVitalRestoreContext(kernel_limits=KernelLimits(max_nodes=512), policy=policy,
    actuator_ids=actuator_ids, embodiment_fingerprint=actuator_context.fingerprint,
    expected_max_concurrent=int(sys.argv[2]),
    competence_gate=CompetenceGate(min_support=3, min_consistency=0.8),
    running_version=(0, 85, 0))
restored = restore_vital_state(transition.owner_checkpoint, context)
assert restored.transition.to_bytes() == transition.to_bytes()
restored_runtime = OrganismVitalRuntime.from_restored_state(restored)
restored_checkpoint = restored_runtime.accepted_source_checkpoint
restored_aggregate = OrganismLongitudinalCheckpoint.from_organism_checkpoint(restored_checkpoint)
actual_digests = {name: sha256(payload).hexdigest()
    for name, payload in restored_aggregate.components.items()}
assert actual_digests == json.load(open(sys.argv[3], encoding='utf-8'))
assert restored.cognition.sensory_normalizers['sense'].count == 2
assert restored.experience.max_records == 64
assert restored.agency.checkpoint()['query_budget'] == 5
assert restored.competencies.checkpoint()['competence_gate'] == {'min_support': 3, 'min_consistency': 0.8}
competency = json.loads(OrganismLongitudinalCheckpoint.from_organism_checkpoint(
    transition.owner_checkpoint).components['competencies'])
assert restored.competence_library.checkpoint() == competency['library']
assert restored.competence_library.checkpoint()['max_competences'] == 17
assert restored.causal_evidence.checkpoint() == competency['causal_evidence']
assert len(restored.causal_evidence.evidence) == 2
assert restored.effect_model.checkpoint() == competency['effect_model']
assert restored.effect_model.checkpoint()['max_contexts'] == 23
assert restored.controllability_model.checkpoint() == competency['controllability_model']
assert restored.agency_model.checkpoint() == competency['agency_model']
print('fresh-process-restore-ok')
"""
    result = subprocess.run([sys.executable, "-c", script, str(payload_path), str(expected_max_concurrent), str(digest_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fresh-process-restore-ok"
    mismatched = subprocess.run(
        [sys.executable, "-c", script, str(payload_path), str(expected_max_concurrent + 1), str(digest_path)],
        capture_output=True, text=True, check=False,
    )
    assert mismatched.returncode != 0
    assert "failed to restore" in mismatched.stderr

def test_checkpoint_requires_explicit_classification_for_every_component() -> None:
    checkpoint = _checkpoint()
    incomplete = dict(checkpoint.classifications)
    incomplete.pop("experience_memory")
    with pytest.raises(InvalidLongitudinalCheckpoint, match="classifications"):
        OrganismLongitudinalCheckpoint(
            checkpoint.identity, checkpoint.components, incomplete
        )


def test_vital_runtime_factory_accepts_only_a_verified_restore_result() -> None:
    from organism.longitudinal import OrganismVitalRuntime

    with pytest.raises(TypeError, match="RestoredOrganismVitalState"):
        OrganismVitalRuntime.from_restored_state(object())


def test_checkpoint_rejects_inactive_declaration_without_verified_profile_proof() -> None:
    with pytest.raises(InvalidLongitudinalCheckpoint, match="inactive components"):
        LongitudinalComponentClassification(
            LongitudinalComponentActivity.INACTIVE,
            "scenario-evidence:genetics-disabled",
        )


def test_missing_active_component_cannot_be_omitted_or_skipped_on_restore() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(InvalidLongitudinalCheckpoint, match="inactive components"):
        LongitudinalComponentClassification(
            LongitudinalComponentActivity.INACTIVE,
            "scenario-evidence:genetics-disabled",
        )

    components = dict(checkpoint.components)
    components.pop("genetic_expression")
    with pytest.raises(InvalidLongitudinalCheckpoint, match="payload mismatch"):
        OrganismLongitudinalCheckpoint(
            checkpoint.identity, components, checkpoint.classifications
        )


def _restore_context(**overrides):
    from organism.agency.policy import ProspectivePolicy
    from organism.cognition.limits import KernelLimits
    from organism.longitudinal import OrganismVitalRestoreContext

    identity = str(_checkpoint().identity)
    values = dict(
        kernel_limits=KernelLimits(max_nodes=512),
        policy=ProspectivePolicy(organism_id=identity),
        actuator_ids=("actuator.0",),
        embodiment_fingerprint="embodiment.test",
        expected_max_concurrent=2,
    )
    values.update(overrides)
    return OrganismVitalRestoreContext(**values)


def test_typed_restore_rejects_wrong_owner_identity_before_materialization() -> None:
    checkpoint = _checkpoint()
    wrong_owner = OrganismCheckpoint(
        OrganismIdentity(UUID("00000000-0000-0000-0000-000000000042")),
        checkpoint.to_organism_checkpoint().private_state,
    )
    from organism.longitudinal import restore_vital_state

    with pytest.raises(InvalidLongitudinalCheckpoint, match="failed to restore"):
        restore_vital_state(wrong_owner, _restore_context())


def test_typed_restore_rejects_corrupt_component_evidence_before_materialization() -> None:
    checkpoint = _checkpoint()
    components = dict(checkpoint.components)
    components["cognitive_graph"] += b"tampered"
    tampered = OrganismLongitudinalCheckpoint(
        checkpoint.identity, components, checkpoint.classifications
    )
    owner = OrganismCheckpoint(checkpoint.identity, tampered.to_bytes())
    from organism.longitudinal import restore_vital_state

    with pytest.raises(InvalidLongitudinalCheckpoint, match="failed to restore"):
        restore_vital_state(owner, _restore_context())


def test_typed_restore_rejects_unavailable_dependency_before_materialization() -> None:
    checkpoint = _checkpoint()
    dependency = CheckpointDependency("model://required", sha256(b"required").hexdigest())
    aggregate = OrganismLongitudinalCheckpoint(
        checkpoint.identity, checkpoint.components, checkpoint.classifications,
        dependencies=(dependency,),
    )
    owner = OrganismCheckpoint(
        checkpoint.identity, aggregate.to_bytes(), dependencies=(dependency,),
    )
    from organism.longitudinal import restore_vital_state

    with pytest.raises(InvalidLongitudinalCheckpoint, match="dependency reader is required"):
        restore_vital_state(owner, _restore_context())

