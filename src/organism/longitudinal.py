"""Typed composition boundary for the Owner-approved M2 vital state.

This module deliberately composes existing subsystem-owned codecs. It does
not let callers supply an arbitrary component/encoder registry and does not
claim that a bounded exploration checkpoint is a complete runtime checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Callable, Mapping, Sequence

from .actuation.binding import CompetenceExecutionBindingRegistry
from .actuation.competence import CompetenceLibrary
from .actuation.domain import ActionDomain
from .actuation.evidence import CausalEvidenceLedger
from .actuation.action_history import DependencyDeclaration
from .actuation.model import AgencyModel, CompetenceEffectModel, ControllabilityModel
from .actuation.sensorimotor import CompetenceDevelopmentEngine, CompetenceGate
from .agency.prospective import ProspectiveAgency, ProspectivePolicy
from .checkpoint import (
    CheckpointDependency,
    InvalidLongitudinalCheckpoint,
    LongitudinalComponentActivity,
    LongitudinalComponentClassification,
    OrganismLongitudinalCheckpoint,
    OrganismCheckpoint,
    REQUIRED_LONGITUDINAL_COMPONENTS,
)
from .cognition.generative.registry import GenerativeModelRegistry
from .cognition.limits import KernelLimits
from .cognition.portable_checkpoint import CognitiveRuntimeCheckpoint
from .identity import OrganismIdentity
from .memory.ledger import ExperienceLedger
from .transition import OrganismTransitionState


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _decode_cognition(payload: bytes) -> dict[str, object]:
    """Decode the canonical cognitive owner payload without changing its meaning."""
    try:
        doc = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidLongitudinalCheckpoint("cognitive owner payload is malformed") from exc
    required = {
        "schema", "schema_version", "dependency_closure", "kernel_limits", "identity",
        "transition", "graph", "experience_records", "safety_state",
        "sensory_normalizers", "generative_resident", "genome", "germline_state",
        "expression_state",
    }
    if not isinstance(doc, dict) or set(doc) != required:
        raise InvalidLongitudinalCheckpoint("cognitive owner payload has an invalid shape")
    if doc["schema"] != "biopeia.organism.cognitive-runtime" or type(doc["schema_version"]) is not int:
        raise InvalidLongitudinalCheckpoint("unsupported cognitive owner payload")
    return doc


@dataclass(frozen=True, slots=True)
class OrganismVitalState:
    """The concrete set of owner objects that jointly define continuation.

    Cognitive categories are intentionally sourced from the single canonical
    CognitiveRuntimeCheckpoint rather than independently fabricated flags.
    The separate experience ledger, agency, action domain, competence engine,
    binding registry and transition are captured by their owning codecs.
    """

    transition: OrganismTransitionState
    cognition: CognitiveRuntimeCheckpoint
    experience: ExperienceLedger
    agency: ProspectiveAgency
    competencies: CompetenceDevelopmentEngine
    actions: ActionDomain
    bindings: CompetenceExecutionBindingRegistry
    competence_library: CompetenceLibrary
    causal_evidence: CausalEvidenceLedger
    effect_model: CompetenceEffectModel
    controllability_model: ControllabilityModel
    agency_model: AgencyModel

    def __post_init__(self) -> None:
        identity = self.transition.identity
        if self.cognition.state.identity != identity:
            raise InvalidLongitudinalCheckpoint("transition and cognition identity differ")
        if self.experience.organism_id != str(identity):
            raise InvalidLongitudinalCheckpoint("experience ledger identity differs")
        if self.actions.organism_id != str(identity):
            raise InvalidLongitudinalCheckpoint("action domain identity differs or is absent")
        if self.bindings.organism_id != str(identity):
            raise InvalidLongitudinalCheckpoint("binding registry identity differs or is absent")
        # Verify each subsystem's serialized identity rather than trusting only
        # the outer envelope to attest to independently-owned payloads.
        for label, payload in (
            ("agency", _canonical_json(self.agency.checkpoint())),
            ("competencies", _canonical_json(self.competencies.checkpoint())),
        ):
            try:
                owner_id = json.loads(payload)["organism_id"]
            except (TypeError, KeyError, json.JSONDecodeError) as exc:
                raise InvalidLongitudinalCheckpoint(f"{label} checkpoint has no identity") from exc
            if owner_id != str(identity):
                raise InvalidLongitudinalCheckpoint(f"{label} identity differs")
        cognitive = _decode_cognition(self.cognition.to_bytes())
        # These are mandatory in the approved M2 closure. Do not call absent
        # state inactive merely because the legacy bounded cognition codec
        # allows None/empty defaults.
        if cognitive["safety_state"] is None:
            raise InvalidLongitudinalCheckpoint("learning_safety state is absent")
        if cognitive["genome"] is None or cognitive["germline_state"] is None or cognitive["expression_state"] is None:
            raise InvalidLongitudinalCheckpoint("genetic_expression state is absent")
        if not cognitive["sensory_normalizers"]:
            raise InvalidLongitudinalCheckpoint("sensory_normalizers state is absent")

    @property
    def identity(self) -> OrganismIdentity:
        return self.transition.identity

    def capture(self, *, dependencies: tuple[CheckpointDependency, ...] = ()) -> OrganismCheckpoint:
        """Capture the typed live composition through subsystem-owned codecs."""
        cognition = self.cognition.to_bytes()
        cognitive = _decode_cognition(cognition)
        # The aggregate retains the original canonical owner payload for exact
        # restoration, while each required domain projection is distinct and
        # content-derived from that payload (never a copy mislabeled as a
        # separate component).
        cognition_identity = {
            "schema": cognitive["schema"],
            "schema_version": cognitive["schema_version"],
            "identity": cognitive["identity"],
            "kernel_limits": cognitive["kernel_limits"],
            "dependency_closure": cognitive["dependency_closure"],
        }
        cognition_graph = {**cognition_identity, "graph": cognitive["graph"]}
        cognition_genetics = {
            **cognition_identity,
            "genome": cognitive["genome"],
            "germline_state": cognitive["germline_state"],
            "expression_state": cognitive["expression_state"],
        }
        cognition_safety = {**cognition_identity, "safety_state": cognitive["safety_state"]}
        cognition_normalizers = {
            **cognition_identity,
            "sensory_normalizers": cognitive["sensory_normalizers"],
        }
        experience = {
            "ledger": json.loads(self.experience.to_bytes()),
            "cognitive_records": cognitive["experience_records"],
        }
        lifecycle = {
            "identity": str(self.identity),
            "organism_step": self.transition.organism_step,
            "embodiment": None if self.transition.embodiment is None else {
                "body_id": self.transition.embodiment.body_id,
                "embodiment_episode_id": self.transition.embodiment.embodiment_episode_id,
                "morphology_revision": self.transition.embodiment.morphology_revision,
            },
            "world_correlation": None if self.transition.world_correlation is None else {
                "world_id": self.transition.world_correlation.world_id,
                "causal_tick": self.transition.world_correlation.causal_tick,
            },
            "lifecycle": self.transition.lifecycle.value,
            "lifecycle_history": [{
                "transition_id": item.transition_id,
                "from_state": item.from_state.value,
                "to_state": item.to_state.value,
                "request_ref": item.request_ref,
            } for item in self.transition.lifecycle_history],
        }
        components = {
            # Lifecycle metadata is encoded independently. The outer owner
            # checkpoint below carries the complete aggregate, not a nested
            # cognition-only checkpoint masquerading as vital state.
            "lifecycle": _canonical_json(lifecycle),
            # Keep the complete canonical cognition owner payload in its own
            # codec envelope. The smaller projections below are independently
            # checked against it during restore.
            "cognitive_transition": cognition,
            "cognitive_graph": _canonical_json(cognition_graph),
            "experience_memory": _canonical_json(experience),
            "genetic_expression": _canonical_json(cognition_genetics),
            "learning_safety": _canonical_json(cognition_safety),
            "sensory_normalizers": _canonical_json(cognition_normalizers),
            "agency": _canonical_json(self.agency.checkpoint()),
            "competencies": _canonical_json({
                "engine": self.competencies.checkpoint(),
                "bindings": self.bindings.checkpoint(),
                "library": self.competence_library.checkpoint(),
                "causal_evidence": self.causal_evidence.checkpoint(),
                "effect_model": self.effect_model.checkpoint(),
                "controllability_model": self.controllability_model.checkpoint(),
                "agency_model": self.agency_model.checkpoint(),
            }),
            "action_commitments": self.actions.to_bytes(),
        }
        if set(components) != REQUIRED_LONGITUDINAL_COMPONENTS:
            raise InvalidLongitudinalCheckpoint("typed vital-state closure is incomplete")
        # The reference names the exact captured bytes, so it is independently
        # checkable and cannot be mistaken for a general promise that a codec
        # exists. The complete required set is populated with live owner state.
        classifications = {
            name: LongitudinalComponentClassification(
                LongitudinalComponentActivity.ACTIVE,
                f"component:{name}:sha256:{sha256(components[name]).hexdigest()}",
            )
            for name in sorted(REQUIRED_LONGITUDINAL_COMPONENTS)
        }
        aggregate = OrganismLongitudinalCheckpoint(
            self.identity, components, classifications, dependencies
        )
        # World receives this opaque owner checkpoint. Its private payload is
        # the complete closure; a consumer must not mistake the old cognitive
        # transition codec for a complete vital-state checkpoint.
        return OrganismCheckpoint(self.identity, aggregate.to_bytes(), dependencies)

    def capture_transition(self, *, dependencies: tuple[CheckpointDependency, ...] = ()) -> OrganismTransitionState:
        """Return the normal World envelope carrying the complete M2 closure."""
        return OrganismTransitionState(
            owner_checkpoint=self.capture(dependencies=dependencies),
            organism_step=self.transition.organism_step,
            embodiment=self.transition.embodiment,
            world_correlation=self.transition.world_correlation,
            lifecycle=self.transition.lifecycle,
            lifecycle_history=self.transition.lifecycle_history,
        )


class OrganismVitalRuntime:
    """Owner-held, single-source composition boundary for candidate export.

    The runtime captures one accepted full vital composition at construction.
    Candidate export requires an explicit complete staged owner composition.
    The runtime binds that composition to the produced transition and cognition
    bytes; it never fills missing candidate owners from its accepted baseline.
    Owner objects are mutable, so every export recomputes the canonical full
    checkpoint and refuses if the accepted or pending candidate changes outside
    its corresponding commit boundary.

    This class does not confirm World commits. A pending candidate remains
    pending until this runtime is discarded and recreated from the full state
    selected by World after its own commit verification.
    """

    def __init__(
        self,
        accepted: OrganismVitalState,
        *,
        running_version: tuple[int, int, int] | None = None,
        generative_registry: GenerativeModelRegistry | None = None,
    ) -> None:
        if not isinstance(accepted, OrganismVitalState):
            raise TypeError("accepted must be OrganismVitalState")
        self._accepted = accepted
        # These are explicit runtime configuration, not guessed checkpoint
        # data. They are needed to validate an advanced cognitive candidate.
        self._running_version = running_version
        self._generative_registry = generative_registry
        self._accepted_checkpoint = accepted.capture()
        self._pending_candidate_id: str | None = None
        self._pending_source: bytes | None = None
        self._pending_dependencies: tuple[CheckpointDependency, ...] | None = None
        self._pending_vital_source: bytes | None = None
        self._pending_checkpoint: bytes | None = None
        self._pending_dependency_declaration: DependencyDeclaration | None = None
        self._pending_vital_state: OrganismVitalState | None = None

    @classmethod
    def from_restored_state(
        cls,
        restored: "RestoredOrganismVitalState",
        *,
        running_version: tuple[int, int, int] | None = None,
        generative_registry: GenerativeModelRegistry | None = None,
    ) -> "OrganismVitalRuntime":
        """Create the owner-held runtime directly from the verified restore result.

        This is an integration seam, not a default composition: every required
        owner must have been materialized by ``restore_vital_state`` first.
        Keeping this conversion here avoids callers rebuilding the aggregate
        positionally (and accidentally omitting or substituting an owner).
        """
        if not isinstance(restored, RestoredOrganismVitalState):
            raise TypeError("restored must be RestoredOrganismVitalState")
        accepted = OrganismVitalState(
            transition=restored.transition,
            cognition=restored.cognition,
            experience=restored.experience,
            agency=restored.agency,
            competencies=restored.competencies,
            actions=restored.actions,
            bindings=restored.bindings,
            competence_library=restored.competence_library,
            causal_evidence=restored.causal_evidence,
            effect_model=restored.effect_model,
            controllability_model=restored.controllability_model,
            agency_model=restored.agency_model,
        )
        return cls(
            accepted, running_version=running_version,
            generative_registry=generative_registry,
        )

    @property
    def accepted_source_checkpoint(self) -> OrganismCheckpoint:
        """Return the immutable owner checkpoint that seeded this runtime."""
        return self._accepted_checkpoint

    def candidate_state_for_transition(
        self, transition_state: OrganismTransitionState
    ) -> OrganismVitalState:
        """Compose a full candidate from the accepted owners and produced transition.

        The transition producer owns the cognitive bytes. Other vital owners
        remain the exact instances held by this runtime until their own staging
        APIs are wired into the same cycle. This is intentionally fail-closed:
        it never fills absent components or accepts a caller-built replacement
        composition.
        """
        if not isinstance(transition_state, OrganismTransitionState):
            raise TypeError("transition_state must be OrganismTransitionState")
        if transition_state.identity != self._accepted.identity:
            raise InvalidLongitudinalCheckpoint("candidate transition identity differs from accepted vital state")
        try:
            cognition = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
                transition_state.owner_checkpoint,
                kernel_limits=self._accepted.cognition.graph.kernel_limits,
                generative_registry=self._generative_registry,
                running_version=self._running_version,
            )
        except (TypeError, ValueError) as exc:
            raise InvalidLongitudinalCheckpoint(
                "candidate cognitive state could not be validated under runtime configuration"
            ) from exc
        if cognition.state.identity != self._accepted.identity:
            raise InvalidLongitudinalCheckpoint("candidate cognition identity differs")
        return replace(self._accepted, transition=transition_state, cognition=cognition)

    def checkpoint_for_candidate(
        self,
        composer: object,
        candidate: object,
        *,
        transition_state: OrganismTransitionState,
        candidate_vital_state: OrganismVitalState,
        dependencies: tuple[CheckpointDependency, ...] = (),
    ) -> bytes:
        """Export one staged candidate against the privately held full state.

        Repeated requests for the same candidate/source/dependency tuple are
        idempotent. A different candidate cannot supersede an unresolved one;
        callers must recover from World-selected committed state after a
        rejected or abandoned candidate.
        """
        from dataclasses import replace

        from .actuation.bounded_action import OrganismExplorationComposer

        if not isinstance(composer, OrganismExplorationComposer):
            raise TypeError("composer must be OrganismExplorationComposer")
        if not isinstance(transition_state, OrganismTransitionState):
            raise TypeError("transition_state must be OrganismTransitionState")
        current = self._accepted.capture()
        if current.private_state != self._accepted_checkpoint.private_state:
            raise InvalidLongitudinalCheckpoint(
                "accepted vital state changed outside a committed restore boundary"
            )
        if transition_state.identity != self._accepted.identity:
            raise InvalidLongitudinalCheckpoint("candidate transition identity differs from accepted vital state")
        if not isinstance(candidate_vital_state, OrganismVitalState):
            raise TypeError("candidate_vital_state must be a complete OrganismVitalState")
        if (
            candidate_vital_state.identity != transition_state.identity
            or candidate_vital_state.transition.to_bytes() != transition_state.to_bytes()
            or candidate_vital_state.cognition.to_bytes()
            != transition_state.owner_checkpoint.private_state
        ):
            raise InvalidLongitudinalCheckpoint(
                "candidate vital composition does not match the produced transition and cognition"
            )
        # Every active owner must be supplied explicitly for this candidate.
        # In particular, do not silently substitute accepted memory, agency,
        # binding, or model snapshots when their producers have staged a new
        # value for the same cycle. OrganismVitalState.capture() validates the
        # complete typed composition before it can cross the persistence seam.
        candidate_vital_source = candidate_vital_state.capture().private_state
        if not composer.matches_vital_baseline(
            competencies=candidate_vital_state.competencies,
            actions=candidate_vital_state.actions,
        ):
            raise InvalidLongitudinalCheckpoint(
                "composer baseline differs from explicit candidate competence or action state"
            )

        candidate_id = getattr(candidate, "candidate_id", None)
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate has no stable owner identity")
        source_bytes = transition_state.to_bytes()
        if self._pending_candidate_id is not None:
            if (
                self._pending_candidate_id != candidate_id
                or self._pending_source != source_bytes
                or self._pending_dependencies != dependencies
                or self._pending_vital_source != candidate_vital_source
            ):
                raise InvalidLongitudinalCheckpoint(
                    "another vital-state candidate is unresolved; recover from the last committed cut"
                )
            assert self._pending_checkpoint is not None
            return self._pending_checkpoint

        staged = getattr(composer, "_staged_state", None)
        if staged is None:
            raise InvalidLongitudinalCheckpoint("composer has no staged candidate state")
        staged_engine, staged_actions = staged
        candidate_state = replace(candidate_vital_state, competencies=staged_engine, actions=staged_actions)
        encoded = composer._vital_checkpoint_for_candidate(
            candidate,
            transition_state=transition_state,
            vital_state=candidate_state,
            dependencies=dependencies,
        )
        self._pending_candidate_id = candidate_id
        self._pending_source = source_bytes
        self._pending_dependencies = dependencies
        self._pending_vital_source = candidate_vital_source
        self._pending_checkpoint = encoded
        self._pending_vital_state = candidate_state
        return encoded

    def checkpoint_with_dependency_for_candidate(
        self,
        composer: object,
        candidate: object,
        *,
        transition_state: OrganismTransitionState,
        candidate_vital_state: OrganismVitalState,
        dependencies: tuple[CheckpointDependency, ...] = (),
    ) -> tuple[bytes, DependencyDeclaration | None]:
        """Export the complete vital checkpoint and v5 history declaration.

        The compact exploration codec remains responsible for creating and
        declaring its content-addressed history segment. The persisted owner
        payload, however, is the complete longitudinal composition. World
        stores the declaration's opaque blob before committing that payload.
        v4 composers return no declaration.
        """
        from .actuation.bounded_action import OrganismExplorationComposer

        if not isinstance(composer, OrganismExplorationComposer):
            raise TypeError("composer must be OrganismExplorationComposer")
        declaration = None
        if composer.exports_dependencies:
            _, declaration = composer.checkpoint_with_dependency_for_candidate(
                candidate, transition_state=transition_state
            )
        encoded = self.checkpoint_for_candidate(
            composer, candidate,
            transition_state=transition_state,
            candidate_vital_state=candidate_vital_state,
            dependencies=dependencies,
        )
        if self._pending_dependency_declaration is not None:
            if self._pending_dependency_declaration != declaration:
                raise InvalidLongitudinalCheckpoint(
                    "vital checkpoint dependency declaration changed for pending candidate"
                )
        else:
            self._pending_dependency_declaration = declaration
        return encoded, declaration

    def adopt_after_world_commit(self, composer: object, candidate: object, *, committed_cut: object) -> None:
        """Advance the held full state only for the exact committed candidate.

        The caller is the trusted World composition and must pass its actual
        receipt after commit verification. This validates receipt correlation;
        it does not independently authenticate World provenance.
        """
        from .actuation.bounded_action import CommittedWorldCutReference, OrganismExplorationComposer

        if not isinstance(composer, OrganismExplorationComposer):
            raise TypeError("composer must be OrganismExplorationComposer")
        if not isinstance(committed_cut, CommittedWorldCutReference):
            raise TypeError("adoption requires a World committed-cut receipt")
        if self._pending_vital_state is None or self._pending_candidate_id is None:
            raise InvalidLongitudinalCheckpoint("no complete vital candidate is awaiting commit")
        if getattr(candidate, "candidate_id", None) != self._pending_candidate_id:
            raise InvalidLongitudinalCheckpoint("committed candidate differs from pending vital candidate")
        if (
            getattr(candidate, "world_id", None) != committed_cut.world_id
            or getattr(candidate, "target_causal_tick", None) != committed_cut.causal_tick
            or getattr(candidate, "commit_id", None) != committed_cut.commit_id
        ):
            raise InvalidLongitudinalCheckpoint("committed cut does not match pending vital candidate")
        if self._pending_checkpoint is None or self._pending_dependencies is None:
            raise InvalidLongitudinalCheckpoint(
                "pending vital candidate has no exported checkpoint or dependencies"
            )
        # Vital owner objects are mutable. Re-encode the entire staged
        # composition immediately before adoption and require byte identity
        # with the payload exported to World. Otherwise World could have
        # committed one state while this runtime silently adopts another.
        current_pending = self._pending_vital_state.capture_transition(
            dependencies=self._pending_dependencies
        ).to_bytes()
        if current_pending != self._pending_checkpoint:
            raise InvalidLongitudinalCheckpoint(
                "pending vital state changed after export; recover from the last committed cut"
            )
        if not composer.matches_vital_baseline(
            competencies=self._pending_vital_state.competencies,
            actions=self._pending_vital_state.actions,
        ):
            raise InvalidLongitudinalCheckpoint(
                "composer has not adopted the committed candidate state"
            )
        self._accepted = self._pending_vital_state
        # Keep the accepted baseline exactly equal to the bytes World
        # committed, rather than taking a fresh mutable snapshot.
        self._accepted_checkpoint = OrganismTransitionState.from_bytes(
            self._pending_checkpoint
        ).owner_checkpoint
        self._pending_candidate_id = None
        self._pending_source = None
        self._pending_dependencies = None
        self._pending_vital_source = None
        self._pending_checkpoint = None
        self._pending_dependency_declaration = None
        self._pending_vital_state = None


@dataclass(frozen=True, slots=True)
class OrganismVitalRestoreContext:
    """Non-checkpoint configuration required to materialize runtime owners."""

    kernel_limits: KernelLimits
    policy: ProspectivePolicy
    actuator_ids: Sequence[str]
    embodiment_fingerprint: str
    expected_max_concurrent: int
    competence_gate: CompetenceGate | None = None
    generative_registry: GenerativeModelRegistry | None = None
    running_version: tuple[int, int, int] | None = None
    read_dependency: Callable[[str], bytes] | None = None


@dataclass(frozen=True, slots=True)
class RestoredOrganismVitalState:
    """All-or-nothing typed restoration result; no partial state escapes."""

    transition: OrganismTransitionState
    cognition: CognitiveRuntimeCheckpoint
    experience: ExperienceLedger
    agency: ProspectiveAgency
    competencies: CompetenceDevelopmentEngine
    actions: ActionDomain
    bindings: CompetenceExecutionBindingRegistry
    competence_library: CompetenceLibrary
    causal_evidence: CausalEvidenceLedger
    effect_model: CompetenceEffectModel
    controllability_model: ControllabilityModel
    agency_model: AgencyModel


def restore_vital_state(
    checkpoint: OrganismCheckpoint, context: OrganismVitalRestoreContext
) -> RestoredOrganismVitalState:
    """Restore all active vital owners atomically, rejecting partial closure."""
    if not isinstance(checkpoint, OrganismCheckpoint):
        raise InvalidLongitudinalCheckpoint("expected OrganismCheckpoint")
    if context.read_dependency is None:
        if checkpoint.dependencies:
            raise InvalidLongitudinalCheckpoint("dependency reader is required for referenced state")
        read_dependency = lambda locator: (_ for _ in ()).throw(FileNotFoundError(locator))
    else:
        read_dependency = context.read_dependency
    checkpoint.verify_dependency_closure(read_dependency)
    try:
        aggregate = OrganismLongitudinalCheckpoint.from_bytes(checkpoint.private_state)
        if aggregate.identity != checkpoint.identity:
            raise ValueError("vital state identity mismatch")
        raw = dict(aggregate.components)
        for name, classification in aggregate.classifications.items():
            if classification.evidence_ref != f"component:{name}:sha256:{sha256(raw[name]).hexdigest()}":
                raise ValueError(f"{name} activity evidence does not match captured payload")
        if [
            {"locator": dep.locator, "sha256": dep.sha256_hex} for dep in checkpoint.dependencies
        ] != [
            {"locator": dep.locator, "sha256": dep.sha256_hex} for dep in aggregate.dependencies
        ]:
            raise ValueError("dependency closure mismatch")

        lifecycle = json.loads(raw["lifecycle"])
        expected_lifecycle_fields = {
            "identity", "organism_step", "embodiment", "world_correlation",
            "lifecycle", "lifecycle_history",
        }
        if not isinstance(lifecycle, dict) or set(lifecycle) != expected_lifecycle_fields:
            raise ValueError("invalid lifecycle component")
        cognition = CognitiveRuntimeCheckpoint.from_bytes(
            raw["cognitive_transition"], kernel_limits=context.kernel_limits,
            generative_registry=context.generative_registry,
            running_version=context.running_version,
        )
        if OrganismIdentity.parse(lifecycle["identity"]) != checkpoint.identity or cognition.state.identity != checkpoint.identity:
            raise ValueError("restored owner identity mismatch")
        cognitive = _decode_cognition(cognition.to_bytes())
        cognitive_identity = {
            "schema": cognitive["schema"],
            "schema_version": cognitive["schema_version"],
            "identity": cognitive["identity"],
            "kernel_limits": cognitive["kernel_limits"],
            "dependency_closure": cognitive["dependency_closure"],
        }
        if raw["cognitive_transition"] != cognition.to_bytes():
            raise ValueError("canonical cognition state diverges")
        expected_projections = {
            "cognitive_graph": {**cognitive_identity, "graph": cognitive["graph"]},
            "genetic_expression": {
                **cognitive_identity, "genome": cognitive["genome"],
                "germline_state": cognitive["germline_state"],
                "expression_state": cognitive["expression_state"],
            },
            "learning_safety": {**cognitive_identity, "safety_state": cognitive["safety_state"]},
            "sensory_normalizers": {
                **cognitive_identity, "sensory_normalizers": cognitive["sensory_normalizers"]
            },
        }
        if any(raw[name] != _canonical_json(value) for name, value in expected_projections.items()):
            raise ValueError("cognitive component projections diverge")
        experience_doc = json.loads(raw["experience_memory"])
        if set(experience_doc) != {"ledger", "cognitive_records"} or experience_doc["cognitive_records"] != cognitive["experience_records"]:
            raise ValueError("experience component projections diverge")
        experience = ExperienceLedger.from_bytes(
            _canonical_json(experience_doc["ledger"]), organism_id=str(checkpoint.identity)
        )
        agency_data = json.loads(raw["agency"])
        agency = ProspectiveAgency.restore(
            agency_data, organism_id=str(checkpoint.identity), policy=context.policy
        )
        competency_data = json.loads(raw["competencies"])
        if set(competency_data) != {
            "engine", "bindings", "library", "causal_evidence", "effect_model",
            "controllability_model", "agency_model",
        }:
            raise ValueError("invalid competence component envelope")
        engine = CompetenceDevelopmentEngine.restore(
            competency_data["engine"], actuator_ids=context.actuator_ids,
            organism_id=str(checkpoint.identity),
            embodiment_fingerprint=context.embodiment_fingerprint,
            expected_max_concurrent=context.expected_max_concurrent,
            gate=context.competence_gate,
        )
        bindings = CompetenceExecutionBindingRegistry.from_bytes(
            _canonical_json(competency_data["bindings"])
        )
        library = CompetenceLibrary.from_bytes(
            _canonical_json(competency_data["library"]), gate=context.competence_gate
        )
        causal_evidence = CausalEvidenceLedger.restore(competency_data["causal_evidence"])
        effect_model = CompetenceEffectModel.restore(competency_data["effect_model"])
        controllability_model = ControllabilityModel.restore(competency_data["controllability_model"])
        agency_model = AgencyModel.restore(competency_data["agency_model"])
        actions = ActionDomain.from_bytes(raw["action_commitments"])
        if bindings.organism_id != str(checkpoint.identity):
            raise ValueError("binding registry owner identity mismatch")
        if actions.organism_id != str(checkpoint.identity):
            raise ValueError("action domain owner identity mismatch")
        # Reconstitute the transition envelope around the complete outer
        # checkpoint, so World’s normal owner verifier sees and validates the
        # same complete private payload used by typed restoration.
        from .transition import (
            EmbodimentContextRef, LifecycleTransitionRecord, OrganismLifecycle,
            WorldCausalCorrelation,
        )

        embodiment = lifecycle["embodiment"]
        world_correlation = lifecycle["world_correlation"]
        transition = OrganismTransitionState(
            owner_checkpoint=checkpoint,
            organism_step=lifecycle["organism_step"],
            embodiment=None if embodiment is None else EmbodimentContextRef(**embodiment),
            world_correlation=None if world_correlation is None else WorldCausalCorrelation(**world_correlation),
            lifecycle=OrganismLifecycle(lifecycle["lifecycle"]),
            lifecycle_history=tuple(
                LifecycleTransitionRecord(
                    transition_id=item["transition_id"],
                    from_state=OrganismLifecycle(item["from_state"]),
                    to_state=OrganismLifecycle(item["to_state"]),
                    request_ref=item["request_ref"],
                )
                for item in lifecycle["lifecycle_history"]
            ),
        )
        # Construct once more to apply the same cross-owner identity invariants.
        result = RestoredOrganismVitalState(
            transition, cognition, experience, agency, engine, actions, bindings,
            library, causal_evidence, effect_model, controllability_model, agency_model,
        )
        OrganismVitalState(
            result.transition, result.cognition, result.experience, result.agency,
            result.competencies, result.actions, result.bindings, result.competence_library,
            result.causal_evidence, result.effect_model, result.controllability_model,
            result.agency_model,
        )
        return result
    except InvalidLongitudinalCheckpoint:
        raise
    except Exception as exc:
        raise InvalidLongitudinalCheckpoint("failed to restore complete Organism vital state") from exc
