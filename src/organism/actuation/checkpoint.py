from __future__ import annotations

from typing import Any

from .candidate import ActuatorCandidateState
from .context import ActuatorEvidenceContext
from .proposer import ActuatorEvidenceModel
from .surface import ActuatorConstitution


def export_actuation_state(evidence_model: ActuatorEvidenceModel) -> dict[str, Any]:
    """Serialize established motor-discovery state (spec §11).

    Every candidate's full state round-trips unconditionally — this is the
    organism's own internal checkpoint, not a filtered report to an
    external observer (spec §16.6 rev5). No raw activation/percept
    HISTORY crosses this boundary (PairAccumulator stores bounded Welford
    aggregates, never a sample log), but nothing is withheld by sample
    count.
    """
    return {
        "organism_id": evidence_model.organism_id,
        "context": evidence_model.context.to_payload(),
        "candidates": {state.actuator_id: state.to_payload() for state in evidence_model.states},
    }


def restore_actuation_state(
    payload: dict[str, Any],
    constitution: ActuatorConstitution,
    *,
    organism_id: str,
    context: ActuatorEvidenceContext,
    effect_threshold: float = 0.5,
) -> ActuatorEvidenceModel:
    """Rebuild an ActuatorEvidenceModel from a checkpoint payload.

    Checkpoints are bound to ``organism_id`` and retain the source
    embodiment context. ``payload["candidates"]`` must contain EXACTLY one entry per
    ``constitution.actuator_ids`` — no more, no fewer (spec §15/P0.2).
    ``export_actuation_state`` always exports every candidate, so a payload
    missing one, or one that carries an unknown actuator_id, is a corrupted
    or foreign checkpoint. Silently accepting a partial candidate set would
    let a truncated checkpoint masquerade as "this organism never explored
    some of its actuators" — a plausible-looking but false developmental
    history. Each candidate's own ``actuator_id`` field must also match the
    dict key it is stored under, so a payload cannot smuggle candidate B's
    state in under key A (both being otherwise-known ids).

    ``effect_threshold`` is evidence_model *configuration*, not discovered state,
    and is not part of this payload — the caller supplies construct-
    equivalent config out of band (e.g. from ActuatorConstitution/genome-
    derived defaults). Only discovered candidate state round-trips via the
    payload itself.

    An ``active`` candidate's own recorded evidence must contain a valid
    historical promotion proof (spec §15 — corruption must raise, never
    silently accept an internally-impossible state): promotion is
    exclusively the canonical natural-evidence path (L6.2 — the scheduled
    probing-window promotion route was removed from the organism).

    Restoring onto a different context is an explicit re-embodiment path:
    prior context evidence is retained, but active applicability becomes
    dormant. A new context needs its own evidence before promotion.
    """
    if not isinstance(payload, dict) or set(payload) != {"organism_id", "context", "candidates"}:
        raise ValueError("actuation checkpoint fields do not match the current schema")
    if payload["organism_id"] != organism_id:
        raise ValueError("actuation checkpoint organism_id does not match requested Organism")
    source_context = ActuatorEvidenceContext.from_payload(payload["context"])

    known_ids = set(constitution.actuator_ids)
    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, dict):
        raise ValueError("candidates must be an object")

    payload_ids = set(raw_candidates)
    if payload_ids != known_ids:
        missing = known_ids - payload_ids
        unknown = payload_ids - known_ids
        raise ValueError(
            "checkpoint candidates do not match this organism's ActuatorConstitution — "
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )

    evidence_model = ActuatorEvidenceModel(
        constitution,
        organism_id=organism_id,
        context=context,
        effect_threshold=effect_threshold,
    )
    for actuator_id, raw_state in raw_candidates.items():
        state = ActuatorCandidateState.from_payload(raw_state)
        if state.actuator_id != actuator_id:
            raise ValueError(
                f"checkpoint candidate stored under key {actuator_id!r} but its own "
                f"actuator_id field is {state.actuator_id!r} — key/field mismatch"
            )
        if state.probing_state == "active":
            # NOTE(promotion): Promotion is a historical event. Current cumulative correlation
            # may legitimately weaken after promotion as the organism gathers
            # more experience, so restore must validate the evidence that
            # promotion happened, not require today's effect_strength to still
            # clear the old threshold.
            promotion_context = state.promotion_context_fingerprint
            context_evidence = state.context_evidence.get(promotion_context or "")
            strongest_relation_count = max(
                (
                    relation.count
                    for relation in (
                        context_evidence.effect_relations.values() if context_evidence else ()
                    )
                ),
                default=0,
            )
            natural_min_samples = context_evidence.natural_promotion_samples if context_evidence else 0
            natural_promoted = (
                promotion_context == source_context.fingerprint
                and context_evidence is not None
                and context_evidence.context == source_context
                and natural_min_samples > 0
                and strongest_relation_count >= natural_min_samples
            )

            if not natural_promoted:
                raise ValueError(
                    f"checkpoint candidate {actuator_id!r} claims probing_state='active' but its "
                    "own evidence does not satisfy natural promotion "
                    f"(strongest_relation_count={strongest_relation_count}, "
                    f"natural_min_samples={natural_min_samples}, "
                    f"effect_strength={state.effect_strength}, "
                    f"effect_threshold={effect_threshold})"
                )
        elif state.promotion_context_fingerprint is None and state.natural_promotion_samples != 0:
            raise ValueError("dormant candidate cannot claim promotion samples without context")
        if state.probing_state == "active" and state.promotion_context_fingerprint != context.fingerprint:
            # Keep the previous body's evidence/proof, but never carry current
            # competence into a different body, episode, morphology or surface.
            state.probing_state = "dormant"
            state.natural_promotion_samples = 0
        evidence_model._states[actuator_id] = state  # noqa: SLF001

    return evidence_model
