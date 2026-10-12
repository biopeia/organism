"""Owner-held bounded motor exploration composition.

This path is deliberately distinct from prospective, competence-backed action
intents. It can request a bounded motor command from Organism-owned exploration
state, but makes no claim about purpose, effect, competence, outcome, or learning.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from ..transition import OrganismLifecycle, OrganismTransitionState
from ..cognition.limits import KernelLimits
from ..cognition.portable_checkpoint import CognitiveRuntimeCheckpoint
from ..cognition.generative.registry import GenerativeModelRegistry
from .action import ActionEvaluation, ActionJustification, ActionProposal, ActionSource, MotorCommand
from .context import ActuatorEvidenceContext
from .controller import SequenceController
from .domain import ActionDomain
from .sensorimotor import CompetenceDevelopmentEngine
from .surface import ActuatorSurface
from .types import MotorIntent


@runtime_checkable
class CommittedWorldCutReference(Protocol):
    """Minimal receipt shape from a trusted World composition.

    The concrete receipt remains World-owned. This structural boundary avoids
    an Organism -> World package dependency. It does not authenticate that a
    value was issued by World: the trusted World composition must pass the
    actual World-created receipt, and this boundary only checks its shape and
    correlation with the staged candidate.
    """

    world_id: str
    causal_tick: int
    commit_id: str


def _identifier(kind: str, organism_id: str, world_id: str, tick: int) -> str:
    material = f"{kind}\0{organism_id}\0{world_id}\0{tick}".encode("utf-8")
    return f"{kind}.{sha256(material).hexdigest()[:24]}"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ExplorationActionCandidate:
    """One staged exploration output; it is not adopted until World commits."""

    candidate_id: str
    organism_id: str
    world_id: str
    source_causal_tick: int
    target_causal_tick: int
    physical_step_id: str
    commit_id: str
    base_generation: int
    proposal: ActionProposal | None
    motor_command: MotorCommand | None
    _owner: object


@dataclass(frozen=True, slots=True)
class RestoredOrganismExplorationCheckpoint:
    """State carried by the bounded exploration checkpoint.

    This is not a claim that every Organism-owned subsystem is included in
    the payload or that the complete longitudinal continuation is closed.
    """

    transition_state: OrganismTransitionState
    action_composer: OrganismExplorationComposer


@dataclass(frozen=True, slots=True)
class RestoredOrganismTransitionCheckpoint:
    """Validated materialized view of the supported cognition profile.

    This result covers the serialized Organism transition envelope and its
    self-contained ``CognitiveRuntimeCheckpoint`` payload. It is not a claim
    that agency, other action state, or the complete Organism runtime is
    included. The serialized transition remains the canonical owner state;
    callers of the public transition API continue from that state, which
    validates/restores its embedded payload at the transition boundary.
    """

    transition_state: OrganismTransitionState
    cognitive_runtime: CognitiveRuntimeCheckpoint

    def __post_init__(self) -> None:
        if not isinstance(self.transition_state, OrganismTransitionState):
            raise TypeError("transition_state must be OrganismTransitionState")
        if not isinstance(self.cognitive_runtime, CognitiveRuntimeCheckpoint):
            raise TypeError("cognitive_runtime must be CognitiveRuntimeCheckpoint")
        if self.transition_state.identity != self.cognitive_runtime.state.identity:
            raise ValueError("transition and cognitive runtime identities differ")
        materialized_checkpoint = self.cognitive_runtime.to_organism_checkpoint()
        if materialized_checkpoint.to_bytes() != self.transition_state.owner_checkpoint.to_bytes():
            raise ValueError("cognitive runtime does not match transition checkpoint")


@dataclass(frozen=True, slots=True)
class _DecodedExplorationCheckpoint:
    restored: RestoredOrganismExplorationCheckpoint
    world_id: str
    source_causal_tick: int
    target_causal_tick: int
    physical_step_id: str
    commit_id: str


ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA = "biopeia-organism-exploration-checkpoint-v4"
ORGANISM_TRANSITION_CHECKPOINT_SCHEMA = "organism-transition-v2"


class OrganismCheckpointContract:
    """Owner-side codec contract for World checkpoint validation.

    This adapter validates the two Organism checkpoint forms used by the
    bounded causal-cycle composition: transition state and staged exploration
    state. It does not grant World authority over Organism state, and structural
    verification is not proof that an arbitrary caller is World.
    """

    owner = "organism"

    def __init__(
        self,
        *,
        read_dependency: Callable[[str], bytes] | None = None,
        kernel_limits: KernelLimits | None = None,
    ) -> None:
        self._read_dependency = read_dependency
        # Validation must use the same owner-configured resource ceilings as
        # restoration. The default is the canonical kernel safety profile.
        self._kernel_limits = kernel_limits or KernelLimits()

    def _validate_transition_owner_payload(self, state: OrganismTransitionState) -> None:
        """Validate private owner bytes, not just the generic envelope shape."""
        CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            state.owner_checkpoint,
            kernel_limits=self._kernel_limits,
        )

    def restore_transition_checkpoint(
        self,
        encoded: bytes,
        owner_identity: str,
        *,
        generative_registry: GenerativeModelRegistry | None = None,
        running_version: tuple[int, int, int] | None = None,
    ) -> RestoredOrganismTransitionCheckpoint:
        """Decode and validate the supported cognitive-transition profile before return.

        Decoding and private-state restoration are staged in local values. The
        immutable result is returned only after envelope identity and cognitive
        payload validation both succeed; no partially restored state escapes.
        The result is a validated materialized view; the serialized transition
        remains canonical and is what the public transition path continues from.
        """
        if not isinstance(encoded, bytes):
            raise TypeError("encoded checkpoint must be bytes")
        if len(encoded) > self._kernel_limits.max_owner_checkpoint_bytes:
            raise ValueError("Organism checkpoint exceeds byte limit")
        if not isinstance(owner_identity, str) or not owner_identity.strip():
            raise ValueError("owner_identity must be non-empty")

        transition = OrganismTransitionState.from_bytes(encoded)
        if str(transition.identity) != owner_identity:
            raise ValueError("transition checkpoint does not match expected owner")
        cognitive_runtime = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            transition.owner_checkpoint,
            kernel_limits=self._kernel_limits,
            generative_registry=generative_registry,
            running_version=running_version,
        )
        return RestoredOrganismTransitionCheckpoint(
            transition_state=transition,
            cognitive_runtime=cognitive_runtime,
        )

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        try:
            if (
                not isinstance(encoded, bytes)
                or len(encoded) > self._kernel_limits.max_owner_checkpoint_bytes
            ):
                return False
            if schema_version == ORGANISM_TRANSITION_CHECKPOINT_SCHEMA:
                state = OrganismTransitionState.from_bytes(encoded)
                if str(state.identity) != owner_identity:
                    return False
                self._validate_transition_owner_payload(state)
                if state.owner_checkpoint.dependencies:
                    if self._read_dependency is None:
                        return False
                    state.verify_dependency_closure(self._read_dependency)
                return True
            if schema_version == ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA:
                decoded = OrganismExplorationComposer._decode_checkpoint(
                    encoded,
                    expected_organism_id=owner_identity,
                    read_dependency=self._read_dependency,
                    kernel_limits=self._kernel_limits,
                )
                self._validate_transition_owner_payload(decoded.restored.transition_state)
                return True
        except (TypeError, ValueError):
            return False
        return False

    def verify_restoration(self, reference: object) -> bool:
        """Verify bytes selected by World against its committed reference."""
        if (
            getattr(reference, "owner", None) != self.owner
            or not isinstance(getattr(reference, "owner_identity", None), str)
            or getattr(reference, "schema_version", None)
            not in {ORGANISM_TRANSITION_CHECKPOINT_SCHEMA, ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA}
            or not isinstance(getattr(reference, "locator", None), str)
            or not isinstance(getattr(reference, "sha256", None), str)
        ):
            return False
        try:
            checkpoint_path = Path(reference.locator)
            if checkpoint_path.stat().st_size > self._kernel_limits.max_owner_checkpoint_bytes:
                return False
            encoded = checkpoint_path.read_bytes()
        except OSError:
            return False
        if sha256(encoded).hexdigest() != reference.sha256:
            return False
        return self.validate_checkpoint(
            encoded, reference.owner_identity, reference.schema_version
        )


class OrganismExplorationComposer:
    """Stage and publish bounded exploration outputs behind the Organism owner.

    The owner retains the engine and action-domain state. ``prepare`` works on
    restored copies, so rejection or retry does not mutate accepted state.
    The caller must invoke ``adopt_after_world_commit`` only after the exact
    World cut for this candidate is committed. ``checkpoint_for_candidate``
    exports the staged owner state for the cut; restoration requires a matching
    committed-cut receipt supplied by trusted World code.
    """

    def __init__(
        self,
        *,
        organism_id: str,
        engine: CompetenceDevelopmentEngine,
        action_domain: ActionDomain | None = None,
    ) -> None:
        if not isinstance(organism_id, str) or not organism_id:
            raise ValueError("organism_id must be non-empty")
        if not isinstance(engine, CompetenceDevelopmentEngine):
            raise TypeError("engine must be CompetenceDevelopmentEngine")
        engine_checkpoint = engine.checkpoint()
        stored_actuator_ids = engine_checkpoint.get("actuator_ids")
        stored_organism_id = engine_checkpoint.get("organism_id")
        if (
            not isinstance(stored_actuator_ids, list)
            or not stored_actuator_ids
            or any(not isinstance(value, str) or not value for value in stored_actuator_ids)
            or len(set(stored_actuator_ids)) != len(stored_actuator_ids)
        ):
            raise ValueError("engine must have a valid stable actuator constitution")
        if stored_organism_id != organism_id:
            raise ValueError("engine Organism identity does not match this composer")
        self._organism_id = organism_id
        self._engine = CompetenceDevelopmentEngine.restore(
            engine_checkpoint,
            actuator_ids=tuple(stored_actuator_ids),
            organism_id=organism_id,
            embodiment_fingerprint=engine.embodiment_fingerprint,
        )
        self._domain = (
            ActionDomain.restore(action_domain.checkpoint())
            if action_domain is not None
            else ActionDomain()
        )
        self._generation = 0
        self._staged_candidate_id: str | None = None
        self._staged_state: tuple[CompetenceDevelopmentEngine, ActionDomain] | None = None
        self._staged_transition_digest: str | None = None

    def prepare(
        self,
        transition: OrganismTransitionState,
        *,
        actuator_context: ActuatorEvidenceContext,
        surface: ActuatorSurface,
        target_causal_tick: int,
        physical_step_id: str,
        commit_id: str,
    ) -> ExplorationActionCandidate:
        if not isinstance(transition, OrganismTransitionState):
            raise TypeError("transition must be OrganismTransitionState")
        if str(transition.identity) != self._organism_id:
            raise ValueError("transition Organism identity does not match this composer")
        if transition.lifecycle is not OrganismLifecycle.ACTIVE:
            raise ValueError("exploration requires an ACTIVE Organism transition")
        if transition.embodiment is None or transition.world_correlation is None:
            raise ValueError("exploration requires explicit Embodiment and World context")
        if not isinstance(actuator_context, ActuatorEvidenceContext):
            raise TypeError("actuator_context must be ActuatorEvidenceContext")
        if not isinstance(surface, ActuatorSurface):
            raise TypeError("surface must be ActuatorSurface")

        embodiment = transition.embodiment
        if (
            actuator_context.body_id != embodiment.body_id
            or actuator_context.embodiment_episode_id != embodiment.embodiment_episode_id
            or actuator_context.morphology_revision != embodiment.morphology_revision
        ):
            raise ValueError("actuator applicability does not match the accepted transition")
        if actuator_context.surface_fingerprint != surface.contract_fingerprint:
            raise ValueError("actuator surface fingerprint does not match current applicability")
        if self._engine.embodiment_fingerprint != actuator_context.fingerprint:
            raise ValueError("exploration engine is not bound to current actuator applicability")
        engine_checkpoint = self._engine.checkpoint()
        stored_actuator_ids = engine_checkpoint.get("actuator_ids")
        if (
            not isinstance(stored_actuator_ids, list)
            or any(not isinstance(value, str) or not value for value in stored_actuator_ids)
            or len(set(stored_actuator_ids)) != len(stored_actuator_ids)
            or set(surface.actuator_ids) != set(stored_actuator_ids)
        ):
            raise ValueError("exploration actuator identities do not match the current surface")

        tick = transition.world_correlation.causal_tick
        if type(target_causal_tick) is not int or target_causal_tick != tick + 1:
            raise ValueError(
                "target causal tick must immediately follow the accepted transition source tick"
            )
        if not isinstance(physical_step_id, str) or not physical_step_id.strip():
            raise ValueError("physical_step_id must be non-empty")
        if not isinstance(commit_id, str) or not commit_id.strip():
            raise ValueError("commit_id must be non-empty")
        engine = CompetenceDevelopmentEngine.restore(
            engine_checkpoint,
            # Sensorimotor checkpoint constitution has an internal stable
            # order. Preserve that order; surface collection order is not
            # actuator identity or command authority.
            actuator_ids=tuple(stored_actuator_ids),
            organism_id=self._organism_id,
            embodiment_fingerprint=actuator_context.fingerprint,
        )
        intents = engine.motor_intents(tick)
        if engine.last_output_source != "exploration":
            raise ValueError("learned primitive replay is not admitted as exploration")
        for intent in intents:
            surface.validate(intent.actuator_id, intent.activation)

        domain = ActionDomain.restore(self._domain.checkpoint())
        active = domain.active_commitment
        if active is not None:
            if active.intent_id is not None:
                raise ValueError("exploration cannot replace an active intention commitment")
            domain.close_exploration(
                active.commitment_id,
                tick=tick,
                reason="bounded_exploration_step_complete",
            )

        proposal: ActionProposal | None = None
        command: MotorCommand | None = None
        if intents:
            proposal_id = _identifier("exploration-proposal", self._organism_id, transition.world_correlation.world_id, tick)
            commitment_id = _identifier("exploration-commitment", self._organism_id, transition.world_correlation.world_id, tick)
            controller_id = _identifier("exploration-controller", self._organism_id, transition.world_correlation.world_id, tick)
            command_id = _identifier("exploration-command", self._organism_id, transition.world_correlation.world_id, tick)
            proposal = ActionProposal(
                proposal_id=proposal_id,
                source=ActionSource.EXPLORATION,
                effect_target_id=None,
                competence_id=None,
                justification=ActionJustification(),
                evaluation=ActionEvaluation(
                    effect_confidence=0.0,
                    controllability=None,
                    uncertainty=1.0,
                ),
            )
            controller = SequenceController(
                controller_id=controller_id,
                sequence=(tuple(intents),),
            )
            selection = domain.select_and_commit(
                (proposal,),
                tick=tick,
                commitment_id=commitment_id,
                controller_id=controller_id,
                surface=surface,
                embodiment_id=embodiment.embodiment_episode_id,
                interruptibility=1.0,
                minimum_duration=0,
                maximum_duration=1,
            )
            if selection.commitment is None or selection.decision.proposal is not proposal:
                raise RuntimeError("bounded exploration was not committed")
            command = domain.issue_command(controller, tick=tick, command_id=command_id)

        candidate_id = _identifier(
            f"exploration-candidate-{self._generation}",
            self._organism_id,
            transition.world_correlation.world_id,
            tick,
        )
        candidate = ExplorationActionCandidate(
            candidate_id=candidate_id,
            organism_id=self._organism_id,
            world_id=transition.world_correlation.world_id,
            source_causal_tick=tick,
            target_causal_tick=target_causal_tick,
            physical_step_id=physical_step_id,
            commit_id=commit_id,
            base_generation=self._generation,
            proposal=proposal,
            motor_command=command,
            _owner=self,
        )
        # Keep owner state private. Preparing a newer candidate replaces the
        # previous staged state; only the latest candidate can be adopted.
        self._staged_candidate_id = candidate_id
        self._staged_state = (engine, domain)
        self._staged_transition_digest = sha256(transition.to_bytes()).hexdigest()
        return candidate

    def checkpoint_for_candidate(
        self,
        candidate: ExplorationActionCandidate,
        *,
        transition_state: OrganismTransitionState,
    ) -> bytes:
        """Serialize staged owner state, bound to the exact World commit request.

        These bytes describe a candidate state, not a committed state. World
        must only select this reference after verifying the matching cut.
        """
        if not self._owns_staged_candidate(candidate):
            raise ValueError("candidate is not the currently staged owner state")
        if not isinstance(transition_state, OrganismTransitionState):
            raise TypeError("transition_state must be OrganismTransitionState")
        if sha256(transition_state.to_bytes()).hexdigest() != self._staged_transition_digest:
            raise ValueError("transition state differs from the staged action source")
        correlation = transition_state.world_correlation
        if (
            str(transition_state.identity) != self._organism_id
            or correlation is None
            or correlation.world_id != candidate.world_id
            or correlation.causal_tick != candidate.source_causal_tick
        ):
            raise ValueError("transition state does not match the staged World source cut")
        assert self._staged_state is not None
        engine, domain = self._staged_state
        return self._serialize_checkpoint(
            transition_state=transition_state,
            generation=self._generation + 1,
            engine=engine,
            domain=domain,
            physical_step_id=candidate.physical_step_id,
            commit_id=candidate.commit_id,
        )

    def _serialize_checkpoint(
        self,
        *,
        transition_state: OrganismTransitionState,
        generation: int,
        engine: CompetenceDevelopmentEngine,
        domain: ActionDomain,
        physical_step_id: str,
        commit_id: str,
    ) -> bytes:
        correlation = transition_state.world_correlation
        assert correlation is not None
        payload = {
            "schema": ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
            "organism_id": self._organism_id,
            "generation": generation,
            "world_id": correlation.world_id,
            "source_causal_tick": correlation.causal_tick,
            "target_causal_tick": correlation.causal_tick + 1,
            "physical_step_id": physical_step_id,
            "commit_id": commit_id,
            "transition_state_base64": base64.b64encode(transition_state.to_bytes()).decode("ascii"),
            "engine": engine.checkpoint(),
            "action_domain": domain.checkpoint(),
        }
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def restore_from_committed_checkpoint(
        cls,
        encoded: bytes,
        *,
        committed_cut: CommittedWorldCutReference,
        expected_organism_id: str,
        expected_physical_step_id: str,
        read_dependency: Callable[[str], bytes] | None = None,
        kernel_limits: KernelLimits | None = None,
    ) -> RestoredOrganismExplorationCheckpoint:
        """Restore the checkpoint's bounded state after World cut correlation.

        World remains responsible for verifying that the selected checkpoint
        reference belongs to its committed history. This method only checks
        the owner payload/receipt correlation and refuses uncommitted state.
        It does not claim complete Organism continuation. If the embedded
        transition checkpoint declares external dependencies, this method
        requires and verifies a resolver before returning restored state.
        """
        if not isinstance(committed_cut, CommittedWorldCutReference):
            raise TypeError("restore requires a World committed-cut receipt")
        if not isinstance(expected_organism_id, str) or not expected_organism_id.strip():
            raise ValueError("expected_organism_id must be non-empty")
        if not isinstance(expected_physical_step_id, str) or not expected_physical_step_id.strip():
            raise ValueError("expected_physical_step_id must be non-empty")
        decoded = cls._decode_checkpoint(
            encoded,
            expected_organism_id=expected_organism_id,
            expected_physical_step_id=expected_physical_step_id,
            read_dependency=read_dependency,
            kernel_limits=kernel_limits,
        )
        if (
            decoded.world_id != committed_cut.world_id
            or decoded.target_causal_tick != committed_cut.causal_tick
            or decoded.commit_id != committed_cut.commit_id
        ):
            raise ValueError("exploration checkpoint does not match committed World cut")
        return decoded.restored

    @classmethod
    def _decode_checkpoint(
        cls,
        encoded: bytes,
        *,
        expected_organism_id: str,
        expected_physical_step_id: str | None = None,
        read_dependency: Callable[[str], bytes] | None = None,
        kernel_limits: KernelLimits | None = None,
    ) -> _DecodedExplorationCheckpoint:
        if not isinstance(encoded, bytes):
            raise TypeError("encoded checkpoint must be bytes")
        limits = kernel_limits or KernelLimits()
        if len(encoded) > limits.max_owner_checkpoint_bytes:
            raise ValueError("exploration checkpoint exceeds byte limit")
        if not isinstance(expected_organism_id, str) or not expected_organism_id.strip():
            raise ValueError("expected_organism_id must be non-empty")
        if expected_physical_step_id is not None and (
            not isinstance(expected_physical_step_id, str) or not expected_physical_step_id.strip()
        ):
            raise ValueError("expected_physical_step_id must be non-empty")
        try:
            payload = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid exploration checkpoint encoding") from exc
        expected = {
            "schema", "organism_id", "generation", "world_id",
            "source_causal_tick", "target_causal_tick",
            "physical_step_id", "commit_id", "transition_state_base64", "engine", "action_domain",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("exploration checkpoint fields do not match schema")
        if payload["schema"] != ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported exploration checkpoint schema")
        if payload["organism_id"] != expected_organism_id:
            raise ValueError("exploration checkpoint does not match expected owner or physical step")
        if expected_physical_step_id is not None and payload["physical_step_id"] != expected_physical_step_id:
            raise ValueError("exploration checkpoint does not match expected owner or physical step")
        if (
            not isinstance(payload["organism_id"], str)
            or not payload["organism_id"]
            or type(payload["generation"]) is not int
            or payload["generation"] < 1
            or type(payload["source_causal_tick"]) is not int
            or payload["source_causal_tick"] < 0
            or type(payload["target_causal_tick"]) is not int
            or payload["target_causal_tick"] != payload["source_causal_tick"] + 1
            or not isinstance(payload["world_id"], str)
            or not payload["world_id"]
            or not isinstance(payload["physical_step_id"], str)
            or not payload["physical_step_id"]
            or not isinstance(payload["commit_id"], str)
            or not payload["commit_id"]
            or not isinstance(payload["transition_state_base64"], str)
            or not isinstance(payload["engine"], dict)
            or not isinstance(payload["action_domain"], dict)
        ):
            raise ValueError("invalid exploration checkpoint values")
        try:
            transition_bytes = base64.b64decode(payload["transition_state_base64"], validate=True)
            transition_state = OrganismTransitionState.from_bytes(transition_bytes)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid embedded Organism transition checkpoint") from exc
        if (
            str(transition_state.identity) != payload["organism_id"]
            or transition_state.world_correlation is None
            or transition_state.world_correlation.world_id != payload["world_id"]
            or transition_state.world_correlation.causal_tick != payload["source_causal_tick"]
        ):
            raise ValueError("embedded Organism transition does not match checkpoint context")
        if transition_state.owner_checkpoint.dependencies:
            if read_dependency is None:
                raise ValueError(
                    "owner checkpoint dependencies require a dependency reader"
                )
            transition_state.verify_dependency_closure(read_dependency)
        engine_payload = payload["engine"]
        actuator_ids = engine_payload.get("actuator_ids")
        fingerprint = engine_payload.get("embodiment_fingerprint")
        organism_id = payload["organism_id"]
        if (
            not isinstance(actuator_ids, list)
            or not actuator_ids
            or any(not isinstance(value, str) or not value for value in actuator_ids)
            or not isinstance(fingerprint, str)
            or not fingerprint
        ):
            raise ValueError("invalid exploration engine checkpoint constitution")
        engine = CompetenceDevelopmentEngine.restore(
            engine_payload,
            actuator_ids=tuple(actuator_ids),
            organism_id=organism_id,
            embodiment_fingerprint=fingerprint,
        )
        domain = ActionDomain.restore(payload["action_domain"])
        restored = cls(organism_id=organism_id, engine=engine, action_domain=domain)
        restored._generation = payload["generation"]
        return _DecodedExplorationCheckpoint(
            restored=RestoredOrganismExplorationCheckpoint(transition_state, restored),
            world_id=payload["world_id"],
            source_causal_tick=payload["source_causal_tick"],
            target_causal_tick=payload["target_causal_tick"],
            physical_step_id=payload["physical_step_id"],
            commit_id=payload["commit_id"],
        )

    def _owns_staged_candidate(self, candidate: ExplorationActionCandidate) -> bool:
        return (
            isinstance(candidate, ExplorationActionCandidate)
            and candidate._owner is self
            and candidate.base_generation == self._generation
            and candidate.candidate_id == self._staged_candidate_id
            and self._staged_state is not None
            and self._staged_transition_digest is not None
        )

    def adopt_after_world_commit(
        self,
        candidate: ExplorationActionCandidate,
        *,
        committed_cut: CommittedWorldCutReference,
    ) -> None:
        """Publish staged state for a matching cut supplied by trusted World code.

        The structural receipt check is not proof of World issuance. The caller
        must be the trusted World composition and pass its actual commit receipt;
        this method verifies shape and correlation, not provenance. No World
        integration or end-to-end committed physical step is demonstrated here.
        """
        if not isinstance(candidate, ExplorationActionCandidate) or candidate._owner is not self:
            raise ValueError("candidate was not prepared by this Organism composer")
        if not isinstance(committed_cut, CommittedWorldCutReference):
            raise TypeError("adoption requires a World committed-cut receipt")
        if not isinstance(committed_cut.commit_id, str) or not committed_cut.commit_id.strip():
            raise ValueError("World committed-cut receipt must include a commit identity")
        if candidate.base_generation != self._generation:
            raise ValueError("candidate base is stale; action state already advanced")
        if candidate.candidate_id != self._staged_candidate_id or self._staged_state is None:
            raise ValueError("candidate is no longer the staged Organism action state")
        if (
            candidate.world_id != committed_cut.world_id
            or candidate.target_causal_tick != committed_cut.causal_tick
            or candidate.commit_id != committed_cut.commit_id
        ):
            raise ValueError("candidate does not match the committed World causal cut")
        self._engine, self._domain = self._staged_state
        self._staged_candidate_id = None
        self._staged_state = None
        self._staged_transition_digest = None
        self._generation += 1


__all__ = ["ExplorationActionCandidate", "OrganismExplorationComposer"]
