from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from organism.statistics import PairAccumulator

from .context import ActuatorEvidenceContext
from .types import ActuatorId, _require_nonneg_finite, _require_nonneg_int

ProbingState = Literal["active", "dormant"]
_VALID_PROBING_STATES: frozenset[str] = frozenset({"active", "dormant"})
_MAX_EFFECT_RELATIONS_PER_CANDIDATE = 256
_MAX_CONTEXTS_PER_CANDIDATE = 128


@dataclass(slots=True)
class ActuatorContextEvidence:
    """Evidence and promotion proof accumulated under one embodiment context."""

    context: ActuatorEvidenceContext
    effect_relations: dict[str, PairAccumulator] = field(default_factory=dict)
    natural_promotion_samples: int = 0

    def observe_effect(self, percept_id: str, *, activation: float, delta_percept: float) -> None:
        relation = self.effect_relations.get(percept_id)
        if relation is None:
            if len(self.effect_relations) >= _MAX_EFFECT_RELATIONS_PER_CANDIDATE:
                raise ValueError("context effect_relations exceeds configured bound")
            relation = PairAccumulator()
            self.effect_relations[percept_id] = relation
        relation.observe(activation, delta_percept)

    @property
    def effect_strength(self) -> float:
        strengths = [
            abs(relation.correlation)
            for relation in self.effect_relations.values()
            if relation.correlation is not None
        ]
        return max(strengths, default=0.0)

    def to_payload(self) -> dict[str, Any]:
        return {
            "context": self.context.to_payload(),
            "effect_relations": {
                percept_id: relation.to_payload()
                for percept_id, relation in self.effect_relations.items()
            },
            "natural_promotion_samples": self.natural_promotion_samples,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ActuatorContextEvidence:
        expected = {"context", "effect_relations", "natural_promotion_samples"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("ActuatorContextEvidence payload fields do not match the current schema")
        raw_relations = payload["effect_relations"]
        if not isinstance(raw_relations, dict):
            raise ValueError("context effect_relations must be an object")
        relations = {
            percept_id: PairAccumulator.from_payload(dict(raw))
            for percept_id, raw in raw_relations.items()
        }
        if len(relations) > _MAX_EFFECT_RELATIONS_PER_CANDIDATE:
            raise ValueError("context effect_relations exceeds configured bound")
        return cls(
            context=ActuatorEvidenceContext.from_payload(payload["context"]),
            effect_relations=relations,
            natural_promotion_samples=_require_nonneg_int(
                payload["natural_promotion_samples"], "natural_promotion_samples"
            ),
        )


@dataclass(slots=True)
class ActuatorCandidateState:
    """Bounded, per-actuator record of observed activation -> Δpercept effect.

    Tracks effect/controllability evidence only (spec §5): whether
    activating this channel has a measurable, reproducible consequence.
    Whether that consequence is desirable is never decided here.

    ``effect_relations`` is retained longitudinal aggregate knowledge.
    ``context_evidence`` keeps the same bounded summaries separated by exact
    body/episode/morphology/surface identity; only the current context's
    evidence may promote current applicability. Every field round-trips
    unconditionally. PairAccumulator's ``correlation`` is ``None`` below 3
    samples, so immature relations cannot influence promotion.
    """

    actuator_id: ActuatorId
    activations: int = 0
    effect_relations: dict[str, PairAccumulator] = field(default_factory=dict)
    cost_evidence: float = 0.0
    probing_state: ProbingState = "dormant"
    last_seen_tick: int = 0
    natural_promotion_samples: int = 0
    promotion_context_fingerprint: str | None = None
    context_evidence: dict[str, ActuatorContextEvidence] = field(default_factory=dict)

    def observe_effect(
        self,
        percept_id: str,
        *,
        activation: float,
        delta_percept: float,
    ) -> None:
        """Record one activation/effect pair for ``percept_id``.

        Contract (spec §6 revisión 4): the caller owns computing
        ``delta_percept`` as ``percept(t+1) - percept(t)`` for the SAME
        actuator activation issued at tick ``t`` — i.e. the percept reading
        taken one tick AFTER the activation, minus the reading taken at (or
        immediately before) the activation. This method does not and
        cannot verify that contract itself; it only accumulates whatever
        pair it is handed.

        Passing a same-tick delta (e.g. ``percept(t) - percept(t)``, which
        is ~0 for any stable percept) defeats the entire purpose of this
        accumulator: it manufactures a constant-zero ``y`` series that can
        never show causal covariance with ``activation``, silently masking
        real effect relations as "no detectable effect" regardless of
        whether the actuator is actually causal. Always derive
        ``delta_percept`` from two chronologically distinct percept
        readings taken across the tick boundary that follows the
        activation being recorded.
        """
        self.activations += 1
        relation = self.effect_relations.get(percept_id)
        if relation is None:
            if len(self.effect_relations) >= _MAX_EFFECT_RELATIONS_PER_CANDIDATE:
                self._evict_weakest_relation()
            relation = PairAccumulator()
            self.effect_relations[percept_id] = relation
        relation.observe(activation, delta_percept)

    def observe_context_effect(
        self,
        context: ActuatorEvidenceContext,
        percept_id: str,
        *,
        activation: float,
        delta_percept: float,
    ) -> None:
        context_key = context.fingerprint
        context_evidence = self.context_evidence.get(context_key)
        if context_evidence is None:
            if len(self.context_evidence) >= _MAX_CONTEXTS_PER_CANDIDATE:
                raise ValueError(
                    "actuator context evidence reached its retention bound; refusing to discard history"
                )
            context_evidence = ActuatorContextEvidence(context=context)
            self.context_evidence[context_key] = context_evidence
        elif context_evidence.context != context:
            raise ValueError("context fingerprint collision")
        context_evidence.observe_effect(
            percept_id, activation=activation, delta_percept=delta_percept
        )

    def _evict_weakest_relation(self) -> None:
        def strength(item: tuple[str, PairAccumulator]) -> float:
            correlation = item[1].correlation
            return abs(correlation) if correlation is not None else -1.0

        weakest_id, _ = min(self.effect_relations.items(), key=strength)
        del self.effect_relations[weakest_id]

    @property
    def effect_strength(self) -> float:
        strengths = [
            abs(relation.correlation)
            for relation in self.effect_relations.values()
            if relation.correlation is not None
        ]
        return max(strengths, default=0.0)

    def to_payload(self) -> dict[str, Any]:
        # Every field below is exported UNCONDITIONALLY, exactly as it is —
        # this is a checkpoint, not a filtered report to an external
        # observer (see the class docstring / spec §16.6 rev5).
        return {
            "actuator_id": self.actuator_id,
            "activations": self.activations,
            "cost_evidence": self.cost_evidence,
            "probing_state": self.probing_state,
            "last_seen_tick": self.last_seen_tick,
            "natural_promotion_samples": self.natural_promotion_samples,
            "promotion_context_fingerprint": self.promotion_context_fingerprint,
            "context_evidence": {
                context_key: evidence.to_payload()
                for context_key, evidence in self.context_evidence.items()
            },
            "effect_relations": {
                percept_id: relation.to_payload()
                for percept_id, relation in self.effect_relations.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ActuatorCandidateState":
        required = {
            "actuator_id",
            "activations",
            "cost_evidence",
            "probing_state",
            "last_seen_tick",
            "natural_promotion_samples",
            "promotion_context_fingerprint",
            "context_evidence",
            "effect_relations",
        }
        if set(payload) != required:
            raise ValueError(
                "ActuatorCandidateState payload fields do not match the current schema"
            )

        actuator_id = payload["actuator_id"]
        if not isinstance(actuator_id, str) or not actuator_id:
            raise ValueError("actuator_id must be a non-empty string")

        probing_state = payload["probing_state"]
        if probing_state not in _VALID_PROBING_STATES:
            raise ValueError(
                f"probing_state must be one of {sorted(_VALID_PROBING_STATES)}, got {probing_state!r}"
            )
        raw_relations = payload["effect_relations"]
        if not isinstance(raw_relations, dict):
            raise ValueError("effect_relations must be an object")
        effect_relations = {
            percept_id: PairAccumulator.from_payload(dict(raw))
            for percept_id, raw in raw_relations.items()
        }
        if len(effect_relations) > _MAX_EFFECT_RELATIONS_PER_CANDIDATE:
            raise ValueError(
                "effect_relations exceeds max of "
                f"{_MAX_EFFECT_RELATIONS_PER_CANDIDATE}, got {len(effect_relations)}"
            )

        raw_context_evidence = payload["context_evidence"]
        if not isinstance(raw_context_evidence, dict):
            raise ValueError("context_evidence must be an object")
        if len(raw_context_evidence) > _MAX_CONTEXTS_PER_CANDIDATE:
            raise ValueError("context_evidence exceeds configured retention bound")
        context_evidence = {
            context_key: ActuatorContextEvidence.from_payload(dict(raw_evidence))
            for context_key, raw_evidence in raw_context_evidence.items()
        }
        if any(
            context_key != evidence.context.fingerprint
            for context_key, evidence in context_evidence.items()
        ):
            raise ValueError("context evidence key does not match its identity fingerprint")
        if any(
            len(evidence.effect_relations) > _MAX_EFFECT_RELATIONS_PER_CANDIDATE
            for evidence in context_evidence.values()
        ):
            raise ValueError("context effect_relations exceeds configured bound")

        promotion_context = payload["promotion_context_fingerprint"]
        if promotion_context is not None and (
            not isinstance(promotion_context, str) or not promotion_context
        ):
            raise ValueError("promotion_context_fingerprint must be null or non-empty string")

        return cls(
            actuator_id=actuator_id,
            activations=_require_nonneg_int(payload["activations"], "activations"),
            effect_relations=effect_relations,
            cost_evidence=_require_nonneg_finite(payload["cost_evidence"], "cost_evidence"),
            probing_state=probing_state,  # type: ignore[arg-type]
            last_seen_tick=_require_nonneg_int(payload["last_seen_tick"], "last_seen_tick"),
            natural_promotion_samples=_require_nonneg_int(
                payload["natural_promotion_samples"],
                "natural_promotion_samples",
            ),
            promotion_context_fingerprint=promotion_context,
            context_evidence=context_evidence,
        )
