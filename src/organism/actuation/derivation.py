"""Evidence-backed derivation of MotorCompetence and Execution Binding (#39).

This module connects learned motor primitives and causal evidence into
explicit, audit-traceable MotorCompetence and CompetenceExecutionBinding
records. It strictly honors ratified contracts:
- A body change alone does not establish competence or controllability.
- Derivation requires verified action-linked evidence from CausalEvidenceLedger
  compared against relevant passive (no-intervention) controls.
- Positive causal confidence is never fabricated; if causal contrast is absent
  or non-positive, competence gate and controllability evaluation fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .binding import (
    CompetenceExecutionBinding,
    CompetenceExecutionBindingRegistry,
)
from .competence import (
    CompetenceEvidence,
    CompetenceGate,
    CompetenceLibrary,
    CompetenceMaturity,
    DEFAULT_COMPETENCE_GATE,
    MotorCompetence,
)
from .context import ActuatorEvidenceContext
from .evidence import CausalEvidenceLedger
from .model import (
    CausalSourceKind,
    ControllabilityEstimate,
    ControllabilityModel,
)
from .sensorimotor import MotorPrimitive


@dataclass(frozen=True, slots=True)
class DerivedCompetenceRecord:
    """A verified motor competence derived from primitive and causal evidence."""

    competence: MotorCompetence
    binding: CompetenceExecutionBinding | None
    controllability_estimate: ControllabilityEstimate | None


def derive_motor_competence_from_evidence(
    primitive: MotorPrimitive,
    *,
    effect_id: str,
    ledger: CausalEvidenceLedger,
    controllability_model: ControllabilityModel,
    context_ref: str | None = None,
    applicability_context: ActuatorEvidenceContext | None = None,
    tick: int,
    competence_library: CompetenceLibrary | None = None,
    binding_registry: CompetenceExecutionBindingRegistry | None = None,
    gate: CompetenceGate = DEFAULT_COMPETENCE_GATE,
) -> DerivedCompetenceRecord:
    """Derive one MotorCompetence from action-linked evidence in the ledger.

    Preconditions:
    - ``primitive`` must be an established MotorPrimitive.
    - ``ledger`` must contain positive action-linked opportunities for the primitive
      and a valid non-action/passive baseline.
    - ``context_ref`` selects the exact evidence context; evidence references in
      the result are restricted to that same context.
    - An execution binding is created only when ``applicability_context`` is
      supplied and ``context_ref`` equals its fingerprint. The context carries
      the owner-supplied current body/episode/morphology/surface identity.
    - If causal contrast (action hit rate vs passive hit rate) is zero or negative,
      or if no action hits exist, controllability confidence is 0.0.
    """
    if not isinstance(primitive, MotorPrimitive):
        raise TypeError("primitive must be a MotorPrimitive")
    if not isinstance(ledger, CausalEvidenceLedger):
        raise TypeError("ledger must be a CausalEvidenceLedger")
    if not isinstance(controllability_model, ControllabilityModel):
        raise TypeError("controllability_model must be a ControllabilityModel")
    if applicability_context is not None and not isinstance(
        applicability_context, ActuatorEvidenceContext
    ):
        raise TypeError("applicability_context must be an ActuatorEvidenceContext or None")
    if applicability_context is not None:
        if context_ref != applicability_context.fingerprint:
            raise ValueError(
                "context_ref must match the explicit current Embodiment applicability context"
            )
    if not isinstance(effect_id, str) or not effect_id.strip():
        raise ValueError("effect_id must be a non-empty string")
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
        raise ValueError("tick must be a non-negative integer")

    competence_id = f"competence.{primitive.primitive_id}"
    controller_id = f"controller.{primitive.primitive_id}"

    # 1. Update/query controllability from ledger against passive controls
    estimate = controllability_model.update_from_ledger(
        ledger,
        source_kind=CausalSourceKind.COMPETENCE,
        source_ref=competence_id,
        effect_id=effect_id,
        context_id=context_ref,
        tick=tick,
    )

    # 2. Gather factual evidence references from ledger
    opportunities = ledger.effect_opportunities(
        effect_id,
        competence_id=competence_id,
        context_ref=context_ref,
    )
    action_n, action_hits, other_n, other_hits = opportunities

    # Extract evidence references from the same context used for the estimate.
    evidence_ids: list[str] = [
        item.evidence_id
        for item in reversed(ledger.intervention_evidence)
        if item.competence_id == competence_id
        and item.effect_id == effect_id
        and item.context_ref == context_ref
    ][:8]

    reproducibility = action_hits / action_n if action_n > 0 else 0.0
    failures = action_n - action_hits

    evidence = CompetenceEvidence(
        controller_seed_ref=primitive.primitive_id,
        effect_evidence_refs=(effect_id,),
        context_evidence_refs=((context_ref,) if context_ref else ()),
        controllability_evidence_refs=tuple(reversed(evidence_ids)),
        support=action_hits,
        failures=failures,
        reproducibility=reproducibility,
        controllability=estimate.confidence,
        directional_consistency=primitive.directional_consistency,
        gate=gate,
    )

    competence = MotorCompetence(
        competence_id=competence_id,
        controller_id=controller_id,
        effect_id=effect_id,
        evidence=evidence,
        parent_competence_ids=(),
        controller_strategy_ref=primitive.primitive_id,
    )

    if competence_library is not None:
        competence_library.add(competence)

    # 3. Create execution binding if binding_registry supplied and evidence is valid
    binding: CompetenceExecutionBinding | None = None
    if binding_registry is not None and applicability_context is not None:
        if evidence_ids and competence.maturity in {
            CompetenceMaturity.ESTABLISHED,
            CompetenceMaturity.ROBUST,
        }:
            binding = binding_registry.bind_from_evidence(
                competence_id=competence_id,
                surface_fingerprint=applicability_context.surface_fingerprint,
                applicability_context=applicability_context,
                effect_id=effect_id,
                evidence_refs=tuple(evidence_ids),
                reliability=reproducibility,
                controllability=estimate.confidence,
                tick=tick,
            )

    return DerivedCompetenceRecord(
        competence=competence,
        binding=binding,
        controllability_estimate=estimate,
    )


__all__ = [
    "DerivedCompetenceRecord",
    "derive_motor_competence_from_evidence",
]
