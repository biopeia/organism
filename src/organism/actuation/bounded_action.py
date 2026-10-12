"""Owner-held bounded motor exploration composition.

This path is deliberately distinct from prospective, competence-backed action
intents. It can request a bounded motor command from Organism-owned exploration
state, but makes no claim about purpose, effect, competence, outcome, or learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, runtime_checkable

from ..transition import OrganismLifecycle, OrganismTransitionState
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


@dataclass(frozen=True, slots=True)
class ExplorationActionCandidate:
    """One staged exploration output; it is not adopted until World commits."""

    candidate_id: str
    organism_id: str
    world_id: str
    source_causal_tick: int
    target_causal_tick: int
    base_generation: int
    proposal: ActionProposal | None
    motor_command: MotorCommand | None
    _owner: object


class OrganismExplorationComposer:
    """Stage and publish bounded exploration outputs behind the Organism owner.

    The owner retains the engine and action-domain state. ``prepare`` works on
    restored copies, so rejection or retry does not mutate accepted state.
    The caller must invoke ``adopt_after_world_commit`` only after the exact
    World cut for this candidate is committed. This class is not itself a
    durable continuation envelope; the owner checkpoint-closure work remains
    separately required.
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

    def prepare(
        self,
        transition: OrganismTransitionState,
        *,
        actuator_context: ActuatorEvidenceContext,
        surface: ActuatorSurface,
        target_causal_tick: int,
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
            base_generation=self._generation,
            proposal=proposal,
            motor_command=command,
            _owner=self,
        )
        # Keep owner state private. Preparing a newer candidate replaces the
        # previous staged state; only the latest candidate can be adopted.
        self._staged_candidate_id = candidate_id
        self._staged_state = (engine, domain)
        return candidate

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
        ):
            raise ValueError("candidate does not match the committed World causal cut")
        self._engine, self._domain = self._staged_state
        self._staged_candidate_id = None
        self._staged_state = None
        self._generation += 1


__all__ = ["ExplorationActionCandidate", "OrganismExplorationComposer"]
