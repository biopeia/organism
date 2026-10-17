"""Typed composition boundary for the Owner-approved M2 vital state.

This module deliberately composes existing subsystem-owned codecs. It does
not let callers supply an arbitrary component/encoder registry and does not
claim that a bounded exploration checkpoint is a complete runtime checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Mapping, Sequence

from .actuation.binding import CompetenceExecutionBindingRegistry
from .actuation.domain import ActionDomain
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

    def __post_init__(self) -> None:
        identity = self.transition.identity
        if self.cognition.state.identity != identity:
            raise InvalidLongitudinalCheckpoint("transition and cognition identity differ")
        if self.experience.organism_id != str(identity):
            raise InvalidLongitudinalCheckpoint("experience ledger identity differs")
        # Existing ActionDomain/agency/competence codecs bind their own identity;
        # inspect those serialized owner records rather than guessing fields.
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
        if set(competency_data) != {"engine", "bindings"}:
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
        actions = ActionDomain.from_bytes(raw["action_commitments"])
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
            transition, cognition, experience, agency, engine, actions, bindings
        )
        OrganismVitalState(
            result.transition, result.cognition, result.experience, result.agency,
            result.competencies, result.actions, result.bindings,
        )
        return result
    except InvalidLongitudinalCheckpoint:
        raise
    except Exception as exc:
        raise InvalidLongitudinalCheckpoint("failed to restore complete Organism vital state") from exc
