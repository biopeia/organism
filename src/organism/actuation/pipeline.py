"""Canonical Organism-owned action authority pipeline.

This module composes the explicit action authority path:
  ProspectiveDecision (admission)
    → ActionDomain.admit (ActionIntent)
    → ActionProposal
    → ActionDomain.select_and_commit (ActionCommitment)
    → ActionDomain.issue_command (MotorCommand)
    → ActionAttempt (execution recording)

It strictly honors TARGET contracts without fabricating cognitive readouts,
controllability heuristics, reward semantics, or environmental expectations.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agency.affordance import ActionAffordance
from ..agency.intention import ActionIntent, IntentStatus
from ..agency.prospective import ProspectiveDecision
from ..transition import OrganismLifecycle, OrganismTransitionState
from .action import (
    ActionEvaluation,
    ActionJustification,
    ActionProposal,
    ActionSource,
    MotorCommand,
)
from .attempt import ActionAttempt, attempt_id_for_command
from .commitment import ActionCommitment
from .controller import SequenceController
from .domain import ActionDomain
from .surface import ActuatorSurface


@dataclass(frozen=True, slots=True)
class AdmittedActionPlan:
    """Explicit parameters required to compose an admitted action into execution."""

    decision: ProspectiveDecision
    supporting_affordance: ActionAffordance
    intent_id: str
    proposal_id: str
    commitment_id: str
    controller_id: str
    command_id: str
    intervention_signature_id: str
    interruptibility: float = 1.0
    minimum_duration: int = 0
    maximum_duration: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ProspectiveDecision):
            raise TypeError("decision must be a ProspectiveDecision")
        if not isinstance(self.supporting_affordance, ActionAffordance):
            raise TypeError("supporting_affordance must be an ActionAffordance")
        for field_name in (
            "intent_id",
            "proposal_id",
            "commitment_id",
            "controller_id",
            "command_id",
            "intervention_signature_id",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not 0.0 <= self.interruptibility <= 1.0:
            raise ValueError("interruptibility must be in [0.0, 1.0]")
        if self.minimum_duration < 0:
            raise ValueError("minimum_duration must be non-negative")
        if self.maximum_duration is not None and self.maximum_duration < self.minimum_duration:
            raise ValueError("maximum_duration cannot be less than minimum_duration")


@dataclass(frozen=True, slots=True)
class ExecutedActionPipelineResult:
    """Outcome of connecting an accepted transition to the action authority sequence."""

    intent: ActionIntent
    proposal: ActionProposal
    commitment: ActionCommitment
    motor_command: MotorCommand
    attempt: ActionAttempt


def compose_action_from_admitted_decision(
    transition_state: OrganismTransitionState,
    action_domain: ActionDomain,
    plan: AdmittedActionPlan,
    *,
    controller: SequenceController,
    surface: ActuatorSurface,
) -> ExecutedActionPipelineResult:
    """Execute the canonical sequence: admission → proposal → commitment → command.

    Preconditions:
    - ``transition_state`` must be an ACTIVE Organism with valid EmbodimentContextRef
      and WorldCausalCorrelation.
    - Causal tick is strictly derived from ``transition_state.world_correlation.causal_tick``.
    - Embodiment and morphology contexts are verified against ``transition_state.embodiment``.
    """
    if not isinstance(transition_state, OrganismTransitionState):
        raise TypeError("transition_state must be OrganismTransitionState")
    if transition_state.lifecycle is not OrganismLifecycle.ACTIVE:
        raise ValueError("action pipeline requires an ACTIVE OrganismTransitionState")
    if transition_state.embodiment is None:
        raise ValueError("action pipeline requires an explicit Embodiment context")
    if transition_state.world_correlation is None:
        raise ValueError("action pipeline requires an explicit World causal correlation")
    if not isinstance(action_domain, ActionDomain):
        raise TypeError("action_domain must be an ActionDomain")
    if not isinstance(plan, AdmittedActionPlan):
        raise TypeError("plan must be an AdmittedActionPlan")
    if not isinstance(controller, SequenceController):
        raise TypeError("controller must be a SequenceController")
    if not isinstance(surface, ActuatorSurface):
        raise TypeError("surface must be an ActuatorSurface")

    tick = transition_state.world_correlation.causal_tick
    embodiment_context = transition_state.embodiment

    if plan.supporting_affordance.embodiment_id is not None:
        if plan.supporting_affordance.embodiment_id != embodiment_context.embodiment_episode_id:
            raise ValueError(
                "supporting affordance embodiment_id does not match transition embodiment episode"
            )

    # 1. Admission → ActionIntent
    intent = action_domain.admit(
        plan.decision,
        supporting_affordance=plan.supporting_affordance,
        intent_id=plan.intent_id,
        context_ref=plan.supporting_affordance.context_ref,
        embodiment_id=embodiment_context.embodiment_episode_id,
        tick=tick,
    )

    # 2. ActionProposal constructed from admitted decision and supporting affordance
    proposal = ActionProposal(
        proposal_id=plan.proposal_id,
        source=ActionSource.PROSPECTION,
        effect_target_id=plan.decision.anticipated_effect_id,
        competence_id=plan.decision.competence_id,
        justification=ActionJustification(
            effect_target_id=plan.decision.anticipated_effect_id,
            competence_id=plan.decision.competence_id,
            prediction_id=plan.decision.prediction_ref,
            evidence_refs=plan.decision.origin_refs,
        ),
        evaluation=ActionEvaluation(
            epistemic_relevance=plan.decision.epistemic_relevance,
            homeostatic_relevance=plan.decision.homeostatic_relevance,
            effect_confidence=plan.decision.confidence,
            controllability=plan.supporting_affordance.controllability,
            uncertainty=1.0 - plan.decision.confidence,
        ),
        intent_id=intent.intent_id,
    )

    # 3. Proposal → Commitment (Selection and Commitment)
    selection = action_domain.select_and_commit(
        (proposal,),
        tick=tick,
        commitment_id=plan.commitment_id,
        controller_id=plan.controller_id,
        surface=surface,
        embodiment_id=embodiment_context.embodiment_episode_id,
        interruptibility=plan.interruptibility,
        minimum_duration=plan.minimum_duration,
        maximum_duration=plan.maximum_duration,
    )

    commitment = selection.commitment
    if commitment is None or selection.decision.proposal is not proposal:
        raise RuntimeError("failed to commit admitted action proposal")

    # 4. Commitment → MotorCommand
    command = action_domain.issue_command(
        controller,
        tick=tick,
        command_id=plan.command_id,
    )

    # 5. MotorCommand → ActionAttempt
    attempt = ActionAttempt(
        attempt_id=attempt_id_for_command(command.command_id),
        commitment_id=commitment.commitment_id,
        controller_id=controller.controller_id,
        competence_id=commitment.competence_id,
        intervention_signature_id=plan.intervention_signature_id,
        context_ref=plan.supporting_affordance.context_ref or "default",
        embodiment_id=embodiment_context.embodiment_episode_id,
        surface_fingerprint=surface.contract_fingerprint,
        motor_command_ref=command.command_id,
        actuation_ref=f"actuation.{command.command_id}",
        started_tick=tick,
    )

    return ExecutedActionPipelineResult(
        intent=intent,
        proposal=proposal,
        commitment=commitment,
        motor_command=command,
        attempt=attempt,
    )


@dataclass(frozen=True, slots=True)
class AcceptedActionTransition:
    """Outcome of executing an action transition anchored to an accepted Organism transition."""

    transition_state: OrganismTransitionState
    pipeline_result: ExecutedActionPipelineResult


def execute_canonical_action_transition(
    accepted_cognition_or_state: OrganismTransitionState,
    action_domain: ActionDomain,
    plan: AdmittedActionPlan,
    *,
    controller: SequenceController,
    surface: ActuatorSurface,
) -> AcceptedActionTransition:
    """Execute canonical action pipeline anchored to an accepted Organism transition state.

    Validates that the provided state is an ACTIVE OrganismTransitionState with
    valid Embodiment context and World causal correlation matching the plan's
    supporting affordance, controller, and surface, then invokes the action
    composition pipeline.
    """
    if not isinstance(accepted_cognition_or_state, OrganismTransitionState):
        raise TypeError(
            "accepted_cognition_or_state must be an OrganismTransitionState"
        )
    if accepted_cognition_or_state.lifecycle is not OrganismLifecycle.ACTIVE:
        raise ValueError(
            "canonical action transition requires an ACTIVE OrganismTransitionState"
        )
    if accepted_cognition_or_state.embodiment is None:
        raise ValueError(
            "canonical action transition requires an explicit Embodiment context"
        )
    if accepted_cognition_or_state.world_correlation is None:
        raise ValueError(
            "canonical action transition requires an explicit World causal correlation"
        )

    pipeline_result = compose_action_from_admitted_decision(
        accepted_cognition_or_state,
        action_domain,
        plan,
        controller=controller,
        surface=surface,
    )

    return AcceptedActionTransition(
        transition_state=accepted_cognition_or_state,
        pipeline_result=pipeline_result,
    )


__all__ = [
    "AcceptedActionTransition",
    "AdmittedActionPlan",
    "ExecutedActionPipelineResult",
    "compose_action_from_admitted_decision",
    "execute_canonical_action_transition",
]
