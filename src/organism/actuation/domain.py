"""Organism-owned composition of admitted intentions and motor commands.

This domain joins existing action contracts. It does not choose a new policy,
infer physical outcomes, or own the Embodiment's actuator semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agency.executive_outcome import (
    CausalRevisionState,
    ExecutiveOutcomeLedger,
    ExecutiveOutcomeSample,
    OutcomeClass,
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
from .system import ActuatorSystem
from .types import Actuation


@dataclass(frozen=True, slots=True)
class ActionSelection:
    decision: ArbitrationDecision
    commitment: ActionCommitment | None


class ActionDomain:
    """A single explicitly scoped action execution context for one Organism."""

    def __init__(
        self,
        *,
        arbitrator: ActionArbitrator | None = None,
        actuator_system: ActuatorSystem | None = None,
        outcome_ledger: ExecutiveOutcomeLedger | None = None,
    ) -> None:
        self._arbitrator = arbitrator or ActionArbitrator()
        self._actuator_system = actuator_system or ActuatorSystem()
        self._outcome_ledger = outcome_ledger
        self._intents: dict[str, ActionIntent] = {}
        self._commitments: dict[str, ActionCommitment] = {}
        self._active_commitment_id: str | None = None
        self._closed_commitments: set[str] = set()

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
        """Materialize one already-admitted decision as a pending intent."""
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
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a non-negative causal tick")
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
        current = self.active_commitment
        decision = self._arbitrator.choose(proposals=proposals, current=current, tick=tick)
        if decision.proposal is None:
            return ActionSelection(decision=decision, commitment=current)

        proposal = decision.proposal
        if not proposal.intent_id:
            raise ValueError("selected action proposal must reference a registered intent")
        intent = self._intents.get(proposal.intent_id)
        if intent is None or intent.status is not IntentStatus.PENDING:
            raise ValueError("selected proposal does not reference a pending registered intent")
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
        commitment = self.active_commitment
        if commitment is None or self._commitments.get(commitment.commitment_id) is not commitment:
            raise RuntimeError("no registered active action commitment")
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
        return self._actuator_system.execute_command(command, surface, commitment=commitment)

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
        if status not in {IntentStatus.SATISFIED, IntentStatus.FAILED, IntentStatus.INVALIDATED}:
            raise ValueError("terminal close requires satisfied, failed, or invalidated intent")
        if tick < commitment.started_tick:
            raise ValueError("close tick precedes commitment")
        if outcome is not None:
            if outcome.tick != tick:
                raise ValueError("outcome tick must match the terminal action tick")
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
        }[status]
        commitment.terminate(tick=tick, status=commitment_status, reason=reason)
        intent.transition(status, tick=tick, reason=reason)
        self._active_commitment_id = None
        self._closed_commitments.add(commitment_id)
        if outcome is not None:
            if outcome.outcome_class is OutcomeClass.NEUTRAL:
                return
            assert self._outcome_ledger is not None
            assert intent.anticipated_effect_id is not None
            self._outcome_ledger.record(
                (intent.competence_id, intent.anticipated_effect_id),
                outcome,
                revision=causal_revision,
            )


__all__ = ["ActionDomain", "ActionSelection"]
