"""Organism-owned composition of admitted intentions and motor commands.

This domain joins existing action contracts. It does not choose a new policy,
infer physical outcomes, or own the Embodiment's actuator semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..agency.executive_outcome import (
    CausalRevisionState,
    ExecutiveOutcomeLedger,
    ExecutiveOutcomeSample,
    OutcomeClass,
    classify_outcome,
)
from ..agency.affordance import ActionAffordance
from ..agency.intention import ActionIntent, IntentStatus
from ..agency.prospective import ProspectiveDecision
from .action import ActionProposal, MotorCommand
from .arbitration import ActionArbitrator, ArbitrationDecision
from .commitment import ActionCommitment, CommitmentStatus
from .command import motor_command_from_frame
from .controller import ControllerFrame, SequenceController
from .surface import ActuatorSurface
from .system import _ActuatorSystem
from .types import Actuation


@dataclass(frozen=True, slots=True)
class ActionSelection:
    decision: ArbitrationDecision
    commitment: ActionCommitment | None


class ActionDomain:
    """A single explicitly scoped action execution context for one Organism."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        arbitrator: ActionArbitrator | None = None,
        outcome_ledger: ExecutiveOutcomeLedger | None = None,
    ) -> None:
        self._arbitrator = arbitrator or ActionArbitrator()
        self._actuator_system = _ActuatorSystem()
        self._outcome_ledger = outcome_ledger
        self._intents: dict[str, ActionIntent] = {}
        self._commitments: dict[str, ActionCommitment] = {}
        self._active_commitment_id: str | None = None
        self._closed_commitments: set[str] = set()

    def checkpoint(self) -> dict[str, object]:
        """Export the domain-owned action lifecycle state for exact continuation.

        Controllers and actuator surfaces have separate owners and are not
        serialized here; their identities remain referenced by commitments.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "intents": [item.checkpoint() for item in self._intents.values()],
            "commitments": [item.checkpoint() for item in self._commitments.values()],
            "active_commitment_id": self._active_commitment_id,
            "closed_commitment_ids": sorted(self._closed_commitments),
            "outcome_ledger": (
                self._outcome_ledger.checkpoint() if self._outcome_ledger is not None else None
            ),
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "ActionDomain":
        """Restore strict ActionDomain-owned state; reject incomplete joins."""
        expected = {
            "schema_version",
            "intents",
            "commitments",
            "active_commitment_id",
            "closed_commitment_ids",
            "outcome_ledger",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("action domain checkpoint fields do not match the current schema")
        version = payload["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != cls.SCHEMA_VERSION:
            raise ValueError("unsupported action domain checkpoint schema")

        raw_intents = payload["intents"]
        raw_commitments = payload["commitments"]
        raw_closed = payload["closed_commitment_ids"]
        if not isinstance(raw_intents, list) or not isinstance(raw_commitments, list):
            raise ValueError("action domain intents and commitments must be lists")
        if not isinstance(raw_closed, list) or any(type(value) is not str for value in raw_closed):
            raise ValueError("closed commitment ids must be a list of strings")
        if len(raw_closed) != len(set(raw_closed)):
            raise ValueError("duplicate closed commitment id")

        intents: dict[str, ActionIntent] = {}
        for raw in raw_intents:
            item = ActionIntent.restore(raw)
            if item.intent_id in intents:
                raise ValueError("duplicate action intent id")
            intents[item.intent_id] = item

        commitments: dict[str, ActionCommitment] = {}
        for raw in raw_commitments:
            item = ActionCommitment.restore(raw)
            if item.commitment_id in commitments:
                raise ValueError("duplicate action commitment id")
            commitments[item.commitment_id] = item

        active_id = payload["active_commitment_id"]
        if active_id is not None and type(active_id) is not str:
            raise ValueError("active commitment id must be a string or null")
        active = [item for item in commitments.values() if item.active]
        if len(active) > 1 or (active_id is None) != (not active):
            raise ValueError("active commitment reference does not match commitment state")
        if active_id is not None and (active_id not in commitments or not commitments[active_id].active):
            raise ValueError("active commitment reference is missing or terminal")

        closed_ids = set(raw_closed)
        if not closed_ids <= commitments.keys():
            raise ValueError("closed commitment reference is missing")
        if any(commitments[key].active for key in closed_ids):
            raise ValueError("active commitment cannot be listed as closed")

        for commitment in commitments.values():
            if commitment.intent_id is None:
                continue
            intent = intents.get(commitment.intent_id)
            if intent is None:
                raise ValueError("commitment references a missing intent")
            if (
                intent.competence_id != commitment.competence_id
                or intent.anticipated_effect_id != commitment.effect_target_id
            ):
                raise ValueError("commitment and intent provenance do not match")
            expected_status = {
                CommitmentStatus.ACTIVE: IntentStatus.ACTIVE,
                CommitmentStatus.COMPLETED: IntentStatus.SATISFIED,
                CommitmentStatus.FAILED: IntentStatus.FAILED,
                CommitmentStatus.INVALIDATED: IntentStatus.INVALIDATED,
                CommitmentStatus.INTERRUPTED: IntentStatus.INTERRUPTED,
                CommitmentStatus.INCOMPATIBLE: IntentStatus.INVALIDATED,
            }[commitment.status]
            if intent.status is not expected_status:
                raise ValueError("commitment and intent lifecycle states do not match")
        active_intent_ids = {
            item.intent_id for item in commitments.values() if item.active and item.intent_id
        }
        if any(
            intent.status is IntentStatus.ACTIVE and intent.intent_id not in active_intent_ids
            for intent in intents.values()
        ):
            raise ValueError("active intent has no active commitment")

        raw_ledger = payload["outcome_ledger"]
        if raw_ledger is not None and not isinstance(raw_ledger, Mapping):
            raise ValueError("outcome ledger checkpoint must be a mapping or null")
        ledger = ExecutiveOutcomeLedger.restore(raw_ledger) if raw_ledger is not None else None
        domain = cls(outcome_ledger=ledger)
        domain._intents = intents
        domain._commitments = commitments
        domain._active_commitment_id = active_id
        domain._closed_commitments = closed_ids
        return domain

    @property
    def active_commitment(self) -> ActionCommitment | None:
        if self._active_commitment_id is None:
            return None
        commitment = self._commitments[self._active_commitment_id]
        return commitment if commitment.active else None

    def admit(
        self,
        decision: ProspectiveDecision,
        *,
        supporting_affordance: ActionAffordance,
        intent_id: str,
        context_ref: str | None,
        embodiment_id: str | None,
        tick: int,
    ) -> ActionIntent:
        """Materialize a decision emitted by the trusted Organism admission boundary.

        This checks internal consistency; it does not authenticate caller-provided
        affordance provenance. The canonical cognition/admission composition owns
        that guarantee.
        """
        if not isinstance(decision, ProspectiveDecision):
            raise TypeError("ActionDomain requires an admitted ProspectiveDecision")
        if not isinstance(supporting_affordance, ActionAffordance):
            raise TypeError("ActionDomain requires the current supporting ActionAffordance")
        if decision.supporting_affordance_id != supporting_affordance.affordance_id:
            raise ValueError("admission decision does not reference the supplied affordance")
        if (
            decision.competence_id != supporting_affordance.competence_id
            or decision.anticipated_effect_id != supporting_affordance.anticipated_effect_id
            or decision.prediction_ref != supporting_affordance.prediction_ref
            or decision.confidence != supporting_affordance.prediction_confidence
        ):
            raise ValueError("admission decision does not match its supporting affordance")
        if context_ref != supporting_affordance.context_ref:
            raise ValueError("intent context does not match its supporting affordance")
        if embodiment_id != supporting_affordance.embodiment_id:
            raise ValueError("intent embodiment does not match its supporting affordance")
        if intent_id in self._intents:
            raise ValueError("intent_id is already registered")
        _validate_causal_tick(tick, minimum=0, field="tick")
        intent = ActionIntent(
            intent_id=intent_id,
            competence_id=decision.competence_id,
            anticipated_effect_id=decision.anticipated_effect_id,
            context_ref=context_ref,
            embodiment_id=embodiment_id,
            origin_refs=decision.origin_refs,
            prediction_ref=decision.prediction_ref,
            confidence=decision.confidence,
            epistemic_relevance=decision.epistemic_relevance,
            homeostatic_relevance=decision.homeostatic_relevance,
            supporting_affordance_id=decision.supporting_affordance_id,
            created_tick=tick,
            activated_tick=None,
            last_progress_tick=tick,
            status=IntentStatus.PENDING,
            termination_reason=None,
            admission=decision.admission,
        )
        self._intents[intent_id] = intent
        return intent

    def select_and_commit(
        self,
        proposals: tuple[ActionProposal, ...],
        *,
        tick: int,
        commitment_id: str,
        controller_id: str,
        surface: ActuatorSurface,
        embodiment_id: str | None,
        interruptibility: float,
        minimum_duration: int,
        maximum_duration: int | None,
    ) -> ActionSelection:
        """Arbitrate and create one commitment from explicit execution inputs."""
        _validate_causal_tick(tick, minimum=0, field="tick")
        current = self.active_commitment
        if current is not None:
            _validate_causal_tick(tick, minimum=current.started_tick, field="tick")
        decision = self._arbitrator.choose(proposals=proposals, current=current, tick=tick)
        if decision.proposal is None:
            return ActionSelection(decision=decision, commitment=current)

        proposal = decision.proposal
        if not proposal.intent_id:
            raise ValueError("selected action proposal must reference a registered intent")
        intent = self._intents.get(proposal.intent_id)
        if intent is None or intent.status is not IntentStatus.PENDING:
            raise ValueError("selected proposal does not reference a pending registered intent")
        _validate_causal_tick(tick, minimum=intent.created_tick, field="tick")
        if proposal.competence_id != intent.competence_id:
            raise ValueError("proposal competence does not match its admitted intent")
        if proposal.effect_target_id != intent.anticipated_effect_id:
            raise ValueError("proposal effect does not match its admitted intent")
        if intent.embodiment_id is not None and intent.embodiment_id != embodiment_id:
            raise ValueError("execution embodiment does not match its intent")
        if commitment_id in self._commitments:
            raise ValueError("commitment_id is already registered")
        commitment = ActionCommitment(
            commitment_id=commitment_id,
            proposal_id=proposal.proposal_id,
            effect_target_id=proposal.effect_target_id,
            competence_id=proposal.competence_id,
            started_tick=tick,
            controller_id=controller_id,
            surface_fingerprint=surface.contract_fingerprint,
            embodiment_id=embodiment_id,
            intent_id=intent.intent_id,
            interruptibility=interruptibility,
            minimum_duration=minimum_duration,
            maximum_duration=maximum_duration,
        )
        if current is not None:
            # Existing arbitrator semantics only return a replacement when a
            # proposal wins; make the old lifecycle transition explicit.
            current.terminate(
                tick=tick,
                status=CommitmentStatus.INTERRUPTED,
                reason="replaced_by_selected_proposal",
            )
            old_intent = self._intents.get(current.intent_id or "")
            if old_intent is not None and old_intent.status is IntentStatus.ACTIVE:
                old_intent.transition(
                    IntentStatus.INTERRUPTED,
                    tick=tick,
                    reason="replaced_by_selected_proposal",
                )
        intent.transition(IntentStatus.ACTIVE, tick=tick)
        self._commitments[commitment_id] = commitment
        self._active_commitment_id = commitment_id
        return ActionSelection(decision=decision, commitment=commitment)

    def issue_command(
        self,
        controller: SequenceController,
        *,
        tick: int,
        command_id: str,
    ) -> MotorCommand:
        _validate_causal_tick(tick, minimum=0, field="tick")
        commitment = self.active_commitment
        if commitment is None or self._commitments.get(commitment.commitment_id) is not commitment:
            raise RuntimeError("no registered active action commitment")
        _validate_causal_tick(tick, minimum=commitment.started_tick, field="tick")
        frame = controller.frame(commitment, tick=tick)
        return motor_command_from_frame(
            commitment,
            frame,
            command_id=command_id,
            issued_at_tick=tick,
        )

    def execute(self, command: MotorCommand, surface: ActuatorSurface) -> tuple[Actuation, ...]:
        """Execute only a command bound to this domain's registered active commitment."""
        commitment = self._commitments.get(command.commitment_id)
        if commitment is None or commitment is not self.active_commitment:
            raise RuntimeError("motor command does not reference the registered active commitment")
        if (
            command.controller_id != commitment.controller_id
            or command.competence_id != commitment.competence_id
            or command.surface_fingerprint != commitment.surface_fingerprint
            or command.embodiment_id != commitment.embodiment_id
        ):
            raise RuntimeError("motor command provenance does not match its commitment")
        return self._actuator_system._execute_command(command, surface, commitment=commitment)

    def close(
        self,
        commitment_id: str,
        *,
        status: IntentStatus,
        tick: int,
        reason: str,
        outcome: ExecutiveOutcomeSample | None = None,
        causal_revision: CausalRevisionState | None = None,
    ) -> None:
        """Close a registered active action from an explicit factual outcome."""
        if commitment_id in self._closed_commitments:
            raise ValueError("action commitment has already been closed")
        commitment = self._commitments.get(commitment_id)
        if commitment is None or commitment is not self.active_commitment:
            raise ValueError("cannot close an unregistered or inactive commitment")
        intent = self._intents.get(commitment.intent_id or "")
        if intent is None or intent.status is not IntentStatus.ACTIVE:
            raise RuntimeError("active commitment has no active registered intent")
        if status not in {
            IntentStatus.SATISFIED,
            IntentStatus.FAILED,
            IntentStatus.INVALIDATED,
            IntentStatus.INTERRUPTED,
        }:
            raise ValueError(
                "terminal close requires satisfied, failed, invalidated, or interrupted intent"
            )
        _validate_causal_tick(
            tick,
            minimum=max(commitment.started_tick, intent.last_progress_tick),
            field="tick",
        )
        if outcome is not None:
            if not isinstance(outcome, ExecutiveOutcomeSample):
                raise TypeError("outcome must be an ExecutiveOutcomeSample")
            _validate_causal_tick(outcome.tick, minimum=0, field="outcome.tick")
            if outcome.tick != tick:
                raise ValueError("outcome tick must match the terminal action tick")
            if outcome.reason != reason:
                raise ValueError("outcome reason must match the terminal action reason")
            if outcome.outcome_class is not OutcomeClass.NEUTRAL:
                expected_class = classify_outcome(
                    status=status.value,
                    reason=reason,
                    # A non-neutral terminal sample is only admissible when the
                    # relation was valid at the beginning of the action.
                    binding_valid_at_start=True,
                )
                if outcome.outcome_class is not expected_class:
                    raise ValueError("outcome class contradicts terminal status/reason")
            if outcome.outcome_class is not OutcomeClass.NEUTRAL:
                if self._outcome_ledger is None:
                    raise RuntimeError("outcome evidence supplied without an outcome ledger")
                if intent.anticipated_effect_id is None:
                    raise ValueError("outcome evidence requires a declared anticipated effect")
                if outcome.outcome_class is OutcomeClass.SUPPRESS and causal_revision is None:
                    raise ValueError("suppression outcome requires its causal revision state")
        commitment_status = {
            IntentStatus.SATISFIED: CommitmentStatus.COMPLETED,
            IntentStatus.FAILED: CommitmentStatus.FAILED,
            IntentStatus.INVALIDATED: CommitmentStatus.INVALIDATED,
            IntentStatus.INTERRUPTED: CommitmentStatus.INTERRUPTED,
        }[status]
        # Validate the evidence contract before changing lifecycle state. This
        # prevents an invalid suppression record from stranding a closed action.
        if outcome is not None and outcome.outcome_class is not OutcomeClass.NEUTRAL:
            assert self._outcome_ledger is not None
            assert intent.anticipated_effect_id is not None
            self._outcome_ledger.record(
                (intent.competence_id, intent.anticipated_effect_id),
                outcome,
                revision=causal_revision,
            )
        commitment.terminate(tick=tick, status=commitment_status, reason=reason)
        intent.transition(status, tick=tick, reason=reason)
        self._active_commitment_id = None
        self._closed_commitments.add(commitment_id)


def _validate_causal_tick(value: object, *, minimum: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer causal tick >= {minimum}")


__all__ = ["ActionDomain", "ActionSelection"]
