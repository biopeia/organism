"""Predictive, controllability and agency views over canonical causal evidence.

Controllability and agency are inferred for a generalized causal source
(Agency Acquisition v1 §21-§23): a provisional intervention family, a learned
ActionDimension or a MotorCompetence.  One implementation serves all three so
the mathematics is never duplicated, and none of them requires a competence
to exist first.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from .evidence import CausalEvidence, CausalEvidenceLedger, Opportunities, OpportunityView



class CausalSourceKind(StrEnum):
    INTERVENTION = "intervention"
    DIMENSION = "dimension"
    COMPETENCE = "competence"


@dataclass(frozen=True, slots=True)
class EffectPrediction:
    prediction_id: str
    competence_id: str
    context_id: str | None
    effect_id: str
    confidence: float
    support: int
    # Availability annotations (Revision Coherence v1 §3.4); None when the
    # prediction comes from competence experience alone.
    prediction_scope: str | None = None
    executable_now: bool | None = None
    admissible_now: bool | None = None


class CompetenceEffectModel:
    """Bounded abstract forward model: competence/context -> learned effect."""

    def __init__(self, *, max_contexts: int = 512) -> None:
        if max_contexts < 1:
            raise ValueError("max_contexts must be positive")
        self._max_contexts = int(max_contexts)
        self._counts: dict[tuple[str, str | None], Counter[str]] = {}

    def observe(self, evidence: CausalEvidence) -> None:
        if evidence.competence_id is None or evidence.effect_id is None:
            return
        key = (evidence.competence_id, evidence.context_ref)
        counter = self._counts.setdefault(key, Counter())
        counter[evidence.effect_id] += 1
        if len(self._counts) > self._max_contexts:
            retained = sorted(
                self._counts.items(),
                key=lambda item: (-sum(item[1].values()), item[0][0], item[0][1] or ""),
            )[: self._max_contexts]
            self._counts = dict(retained)

    def rebuild(self, ledger: CausalEvidenceLedger) -> None:
        self._counts = {}
        for evidence in ledger.evidence:
            self.observe(evidence)

    @property
    def context_count(self) -> int:
        return len(self._counts)

    def predict(
        self,
        *,
        competence_id: str,
        context_id: str | None,
    ) -> EffectPrediction | None:
        counter = self._counts.get((competence_id, context_id))
        if not counter:
            # Fall back to competence-wide evidence when this context is new.
            merged: Counter[str] = Counter()
            for (known_competence, _known_context), values in self._counts.items():
                if known_competence == competence_id:
                    merged.update(values)
            counter = merged
        if not counter:
            return None
        effect_id, support = min(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        total = sum(counter.values())
        confidence = support / max(1, total)
        material = (f"{competence_id}|{context_id or '*'}|{effect_id}|{support}|{total}").encode(
            "utf-8"
        )
        return EffectPrediction(
            prediction_id="prediction." + hashlib.sha256(material).hexdigest()[:24],
            competence_id=competence_id,
            context_id=context_id,
            effect_id=effect_id,
            confidence=confidence,
            support=support,
        )

    SCHEMA_VERSION = 1

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_contexts": self._max_contexts,
            "contexts": [
                {
                    "competence_id": competence_id,
                    "context_ref": context_ref,
                    "counts": dict(counts),
                }
                for (competence_id, context_ref), counts in sorted(
                    self._counts.items(),
                    key=lambda item: (item[0][0], item[0][1] or ""),
                )
            ],
        }

    @classmethod
    def restore(cls, payload: object) -> "CompetenceEffectModel":
        if not isinstance(payload, dict):
            raise ValueError("invalid competence effect model checkpoint")
        expected_fields = {"schema_version", "max_contexts", "contexts"}
        if set(payload) != expected_fields:
            raise ValueError("invalid competence effect model checkpoint fields")
        version = payload["schema_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != cls.SCHEMA_VERSION
        ):
            raise ValueError("unsupported competence effect model checkpoint version")
        max_contexts = payload["max_contexts"]
        if (
            isinstance(max_contexts, bool)
            or not isinstance(max_contexts, int)
            or max_contexts < 1
        ):
            raise ValueError("invalid max_contexts in checkpoint")
        raw_contexts = payload["contexts"]
        if not isinstance(raw_contexts, list) or len(raw_contexts) > max_contexts:
            raise ValueError("invalid or unbounded contexts list in checkpoint")
        model = cls(max_contexts=max_contexts)
        for entry in raw_contexts:
            if not isinstance(entry, dict) or set(entry) != {
                "competence_id",
                "context_ref",
                "counts",
            }:
                raise ValueError("invalid context entry in checkpoint")
            competence_id = entry["competence_id"]
            context_ref = entry["context_ref"]
            if not isinstance(competence_id, str) or not competence_id:
                raise ValueError("invalid competence_id in context entry")
            if context_ref is not None and (
                not isinstance(context_ref, str) or not context_ref
            ):
                raise ValueError("invalid context_ref in context entry")
            counts = entry["counts"]
            if not isinstance(counts, dict):
                raise ValueError("invalid counts in context entry")
            counter: Counter[str] = Counter()
            for effect_id, count in counts.items():
                if not isinstance(effect_id, str) or not effect_id:
                    raise ValueError("invalid effect_id in counts")
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 1
                ):
                    raise ValueError("invalid count in counts")
                counter[effect_id] = count
            key = (competence_id, context_ref)
            if key in model._counts:
                raise ValueError("duplicate context key in checkpoint")
            model._counts[key] = counter
        return model


def controllability_source_opportunities(
    ledger: OpportunityView,
    *,
    source_kind: CausalSourceKind,
    source_ref: str,
    effect_id: str,
    context_id: str | None,
    dimension_signature_refs: tuple[str, ...] | None = None,
) -> Opportunities:
    """Return source-specific actions against explicit passive windows.

    Other active sources remain available in the ledger but are not a no-action
    counterfactual. Returned counts are retained-row counts, not independent
    trials; unattributable rows are excluded by the ledger.
    """
    if source_kind is CausalSourceKind.COMPETENCE:
        return ledger.controllability_opportunities(
            effect_id,
            field="competence_id",
            values=(source_ref,),
            context_ref=context_id,
        )
    if source_kind is CausalSourceKind.INTERVENTION:
        return ledger.controllability_opportunities(
            effect_id,
            field="intervention_signature_id",
            values=(source_ref,),
            context_ref=context_id,
        )
    if not dimension_signature_refs:
        raise ValueError("dimension opportunities require its intervention families")
    return ledger.controllability_opportunities(
        effect_id,
        field="intervention_signature_id",
        values=dimension_signature_refs,
        context_ref=context_id,
    )


def agency_source_opportunities(
    ledger: OpportunityView,
    *,
    source_kind: CausalSourceKind,
    source_ref: str,
    effect_id: str,
    context_id: str | None,
    dimension_signature_refs: tuple[str, ...] | None = None,
) -> Opportunities:
    """Preserve Agency's existing intervention-versus-alternatives semantics.

    Controllability's stricter passive-baseline attribution is intentionally
    separate: changing that estimator must not silently redefine Agency.
    """
    if source_kind is CausalSourceKind.COMPETENCE:
        return ledger.effect_opportunities(
            effect_id, competence_id=source_ref, context_ref=context_id
        )
    if source_kind is CausalSourceKind.INTERVENTION:
        return ledger.signature_effect_opportunities(
            effect_id, intervention_signature_id=source_ref, context_ref=context_id
        )
    if not dimension_signature_refs:
        raise ValueError("dimension opportunities require its intervention families")
    return ledger.signatures_effect_opportunities(
        effect_id,
        intervention_signature_ids=dimension_signature_refs,
        context_ref=context_id,
    )


_Key = tuple[CausalSourceKind, str, str, str | None]


def controllability_confidence(
    action_n: int, action_hits: int, other_n: int, other_hits: int
) -> float:
    """Evidence-grounded causal controllability confidence in range [0.0, 1.0].

    This bounded score is not a calibrated probability or scientific causal
    identification. Action and comparison denominators are retained evidence
    rows: source rows for the selected action and exact-context passive windows
    for comparison. The ledger excludes co-issued or overlapping active rows
    that cannot be attributed to one source. Repeated rows can share a
    commitment and are not asserted to be independent trials.

    Specification & Invariants:
    - Input parameters:
      * ``action_n``: Total intervention opportunities observed for the source.
      * ``action_hits``: Successful occurrences of the effect under this source.
      * ``other_n``: Total comparison/alternative windows in the evidence ledger.
      * ``other_hits``: Occurrences of the effect during alternative windows.
    - Zero-evidence floor: If ``action_n <= 0`` or ``action_hits <= 0``, returns
      ``0.0``. Absence of alternatives or passive silence is never evidence of
      control without positive action-linked observation.
    - Reliability: Evaluated as empirical hit rate ``action_hits / action_n``.
    - No comparable passive baseline (``other_n == 0``): causal contrast is not
      estimable, so confidence is zero. Action reliability remains available as
      a descriptive rate but is not causal attribution.
    - Contrasted baseline (``other_n > 0``): Causal advantage is evaluated
      against alternative occurrence rate:
      ``advantage = reliability - (other_hits / other_n)``.
      Support weighting saturates at 8 observations across both intervention and
      alternative evidence:
      ``support = min(1.0, min(action_n, other_n) / 8.0)``.
      Confidence is clamped to ``[0.0, 1.0]``: ``max(0.0, min(1.0, advantage * support))``.
    """
    if action_n <= 0 or action_hits <= 0:
        return 0.0
    if other_n <= 0:
        return 0.0
    reliability = action_hits / action_n
    advantage = reliability - (other_hits / other_n)
    support = min(1.0, min(action_n, other_n) / 8.0)
    return max(0.0, min(1.0, advantage * support))


def agency_confidence(
    action_n: int,
    action_hits: int,
    other_n: int,
    other_hits: int,
    prediction_match: float | None,
) -> float:
    """Contingency + specificity + prediction, zero without positive contrast (§23)."""
    temporal = action_hits / action_n if action_n else 0.0
    other_rate = other_hits / other_n if other_n else 0.0
    specificity = max(0.0, min(1.0, temporal - other_rate))
    if temporal <= 0.0 or specificity <= 0.0:
        return 0.0
    predictive = 0.5 if prediction_match is None else max(0.0, min(1.0, float(prediction_match)))
    support_factor = min(1.0, action_n / 8.0)
    counterfactual_factor = min(1.0, other_n / 8.0)
    return max(
        0.0,
        min(
            1.0,
            min(support_factor, counterfactual_factor)
            * (0.45 * temporal + 0.35 * specificity + 0.20 * predictive),
        ),
    )


def _source_keys(ledger: CausalEvidenceLedger) -> dict[_Key, int]:
    """Every (kind, ref, effect, context) with evidence, and its latest tick."""
    keys: dict[_Key, int] = {}
    for item in ledger.intervention_evidence:
        if item.effect_id is None:
            continue
        for kind, ref in (
            (CausalSourceKind.COMPETENCE, item.competence_id),
            (CausalSourceKind.INTERVENTION, item.intervention_signature_id),
        ):
            if ref is None:
                continue
            key = (kind, ref, item.effect_id, item.context_ref)
            keys[key] = max(keys.get(key, 0), item.observation_tick)
    return keys


def _dimension_keys(
    ledger: CausalEvidenceLedger,
    dimensions: Mapping[str, tuple[str, ...]],
) -> dict[_Key, int]:
    """Context-free (dimension, effect) keys: dimensions are discovered context-free."""
    member_of = {
        signature_ref: dimension_id
        for dimension_id, refs in dimensions.items()
        for signature_ref in refs
    }
    keys: dict[_Key, int] = {}
    for item in ledger.intervention_evidence:
        dimension_id = member_of.get(item.intervention_signature_id or "")
        if dimension_id is None or item.effect_id is None:
            continue
        key = (CausalSourceKind.DIMENSION, dimension_id, item.effect_id, None)
        keys[key] = max(keys.get(key, 0), item.observation_tick)
    return keys


MAX_ESTIMATES = 8192


def _require_int(
    value: object,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def _require_float(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return number


def _require_optional_float(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _require_float(value, field=field, minimum=minimum, maximum=maximum)


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_optional_str(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field=field)


def _require_causal_source_kind(value: object, *, field: str) -> CausalSourceKind:
    if not isinstance(value, (str, CausalSourceKind)):
        raise ValueError(f"{field} must be a valid CausalSourceKind")
    try:
        return CausalSourceKind(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


def _restore_controllability_estimate(raw: object) -> ControllabilityEstimate:
    if not isinstance(raw, Mapping):
        raise ValueError("invalid controllability estimate entry")
    expected_fields = {
        "source_kind",
        "source_ref",
        "effect_id",
        "context_id",
        "confidence",
        "reliability",
        "counterfactual_rate",
        "causal_advantage",
        "action_support",
        "counterfactual_support",
        "last_updated_tick",
    }
    if set(raw) != expected_fields:
        raise ValueError("invalid controllability estimate fields")
    source_kind = _require_causal_source_kind(raw["source_kind"], field="source_kind")
    source_ref = _require_str(raw["source_ref"], field="source_ref")
    effect_id = _require_str(raw["effect_id"], field="effect_id")
    context_id = _require_optional_str(raw["context_id"], field="context_id")
    confidence = _require_float(raw["confidence"], field="confidence", minimum=0.0, maximum=1.0)
    reliability = _require_float(raw["reliability"], field="reliability", minimum=0.0, maximum=1.0)
    counterfactual_rate = _require_optional_float(
        raw["counterfactual_rate"], field="counterfactual_rate", minimum=0.0, maximum=1.0
    )
    causal_advantage = _require_optional_float(
        raw["causal_advantage"], field="causal_advantage", minimum=-1.0, maximum=1.0
    )
    action_support = _require_int(raw["action_support"], field="action_support", minimum=0)
    counterfactual_support = _require_int(
        raw["counterfactual_support"], field="counterfactual_support", minimum=0
    )
    last_updated_tick = _require_int(
        raw["last_updated_tick"], field="last_updated_tick", minimum=0
    )
    return ControllabilityEstimate(
        source_kind=source_kind,
        source_ref=source_ref,
        effect_id=effect_id,
        context_id=context_id,
        confidence=confidence,
        reliability=reliability,
        counterfactual_rate=counterfactual_rate,
        causal_advantage=causal_advantage,
        action_support=action_support,
        counterfactual_support=counterfactual_support,
        last_updated_tick=last_updated_tick,
    )


def _restore_agency_estimate(raw: object) -> AgencyEstimate:
    if not isinstance(raw, Mapping):
        raise ValueError("invalid agency estimate entry")
    expected_fields = {
        "source_kind",
        "source_ref",
        "effect_id",
        "context_id",
        "confidence",
        "temporal_contingency",
        "causal_specificity",
        "prediction_match",
        "support",
        "counterfactual_support",
        "last_updated_tick",
    }
    if set(raw) != expected_fields:
        raise ValueError("invalid agency estimate fields")
    source_kind = _require_causal_source_kind(raw["source_kind"], field="source_kind")
    source_ref = _require_str(raw["source_ref"], field="source_ref")
    effect_id = _require_str(raw["effect_id"], field="effect_id")
    context_id = _require_optional_str(raw["context_id"], field="context_id")
    confidence = _require_float(raw["confidence"], field="confidence", minimum=0.0, maximum=1.0)
    temporal = _require_float(
        raw["temporal_contingency"], field="temporal_contingency", minimum=0.0, maximum=1.0
    )
    specificity = _require_float(
        raw["causal_specificity"], field="causal_specificity", minimum=0.0, maximum=1.0
    )
    prediction_match = _require_optional_float(
        raw["prediction_match"], field="prediction_match", minimum=0.0, maximum=1.0
    )
    support = _require_int(raw["support"], field="support", minimum=0)
    counterfactual_support = _require_int(
        raw["counterfactual_support"], field="counterfactual_support", minimum=0
    )
    last_updated_tick = _require_int(
        raw["last_updated_tick"], field="last_updated_tick", minimum=0
    )
    return AgencyEstimate(
        source_kind=source_kind,
        source_ref=source_ref,
        effect_id=effect_id,
        context_id=context_id,
        confidence=confidence,
        temporal_contingency=temporal,
        causal_specificity=specificity,
        prediction_match=prediction_match,
        support=support,
        counterfactual_support=counterfactual_support,
        last_updated_tick=last_updated_tick,
    )


class _SourceIndexed:
    """Shared per-source lookup, bound and persistence for estimate registries."""

    _estimates: dict
    _by_source: dict
    _estimate_type: type

    def _store(self, key: _Key, estimate) -> None:
        self._estimates[key] = estimate
        self._by_source.setdefault((key[0], key[1]), set()).add(key)
        self._by_kind.setdefault(key[0], set()).add(key)
        if len(self._estimates) > MAX_ESTIMATES + MAX_ESTIMATES // 8:
            self._prune()

    def _prune(self) -> None:
        """Keep the most recently supported relations; bounded organism memory."""
        retained = sorted(
            self._estimates.items(),
            key=lambda item: (
                -item[1].last_updated_tick,
                item[0][0],
                item[0][1],
                item[0][2],
                item[0][3] or "",
            ),
        )[:MAX_ESTIMATES]
        self._estimates = dict(retained)
        self._by_source = {}
        self._by_kind = {}
        for key in self._estimates:
            self._by_source.setdefault((key[0], key[1]), set()).add(key)
            self._by_kind.setdefault(key[0], set()).add(key)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": getattr(self, "SCHEMA_VERSION", 1),
            "estimates": [
                {
                    field: (value.value if isinstance(value, CausalSourceKind) else value)
                    for field, value in (
                        (name, getattr(estimate, name))
                        for name in self._estimate_type.__dataclass_fields__
                    )
                }
                for estimate in self.estimates
            ],
        }

    def restore_estimates(self, payload: object) -> None:
        if isinstance(payload, Mapping):
            schema = payload.get("schema_version")
            expected_schema = getattr(self, "SCHEMA_VERSION", 1)
            if (
                isinstance(schema, bool)
                or not isinstance(schema, int)
                or schema != expected_schema
            ):
                raise ValueError("unsupported causal estimate checkpoint schema")
            if set(payload) != {"schema_version", "estimates"}:
                raise ValueError("invalid causal estimate checkpoint fields")
            raw_items = payload["estimates"]
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raise ValueError("invalid or unbounded causal estimate table")

        if not isinstance(raw_items, list) or len(raw_items) > MAX_ESTIMATES + MAX_ESTIMATES // 8:
            raise ValueError("invalid or unbounded causal estimate table")

        self._estimates = {}
        self._by_source = {}
        self._by_kind = {}
        for raw in raw_items:
            if self._estimate_type is ControllabilityEstimate:
                estimate = _restore_controllability_estimate(raw)
            elif self._estimate_type is AgencyEstimate:
                estimate = _restore_agency_estimate(raw)
            else:
                if not isinstance(raw, Mapping):
                    raise ValueError("invalid causal estimate")
                values = dict(raw)
                values["source_kind"] = CausalSourceKind(str(values["source_kind"]))
                estimate = self._estimate_type(**values)

            self._store(
                (
                    estimate.source_kind,
                    estimate.source_ref,
                    estimate.effect_id,
                    estimate.context_id,
                ),
                estimate,
            )

    def estimates_for(self, source_kind: CausalSourceKind) -> tuple:
        keys = self._by_kind.get(source_kind, ())
        return tuple(
            self._estimates[key] for key in sorted(keys, key=lambda k: (k[1], k[2], k[3] or ""))
        )


    def __len__(self) -> int:
        return len(self._estimates)

    @property
    def estimates(self) -> tuple:
        return tuple(
            sorted(
                self._estimates.values(),
                key=lambda item: (
                    item.source_kind,
                    item.source_ref,
                    item.effect_id,
                    item.context_id or "",
                ),
            )
        )

    def for_source(self, source_kind: CausalSourceKind, source_ref: str) -> tuple:
        keys = self._by_source.get((source_kind, source_ref), ())
        return tuple(self._estimates[key] for key in sorted(keys, key=lambda k: (k[2], k[3] or "")))


@dataclass(frozen=True, slots=True)
class ControllabilityEstimate:
    source_kind: CausalSourceKind
    source_ref: str

    effect_id: str
    context_id: str | None

    confidence: float
    reliability: float

    counterfactual_rate: float | None
    causal_advantage: float | None

    action_support: int
    counterfactual_support: int

    last_updated_tick: int


class ControllabilityModel(_SourceIndexed):
    """Infer control only when intervention evidence beats alternatives."""

    SCHEMA_VERSION = 1
    _estimate_type = ControllabilityEstimate

    @classmethod
    def restore(cls, payload: object) -> "ControllabilityModel":
        model = cls()
        model.restore_estimates(payload)
        return model


    def __init__(self) -> None:
        self._estimates: dict[_Key, ControllabilityEstimate] = {}
        self._by_source: dict[tuple[CausalSourceKind, str], set[_Key]] = {}
        self._by_kind: dict[CausalSourceKind, set[_Key]] = {}

    def update_from_ledger(
        self,
        ledger: OpportunityView,
        *,
        source_kind: CausalSourceKind,
        source_ref: str,
        effect_id: str,
        context_id: str | None,
        tick: int,
        dimension_signature_refs: tuple[str, ...] | None = None,
    ) -> ControllabilityEstimate:
        action_n, action_hits, other_n, other_hits = controllability_source_opportunities(
            ledger,
            source_kind=source_kind,
            source_ref=source_ref,
            effect_id=effect_id,
            context_id=context_id,
            dimension_signature_refs=dimension_signature_refs,
        )
        reliability = action_hits / action_n if action_n else 0.0
        counterfactual_rate = (other_hits / other_n) if other_n else None
        advantage = reliability - counterfactual_rate if counterfactual_rate is not None else None
        confidence = controllability_confidence(action_n, action_hits, other_n, other_hits)
        estimate = ControllabilityEstimate(
            source_kind=source_kind,
            source_ref=source_ref,
            effect_id=effect_id,
            context_id=context_id,
            confidence=confidence,
            reliability=reliability,
            counterfactual_rate=counterfactual_rate,
            causal_advantage=advantage,
            action_support=action_n,
            counterfactual_support=other_n,
            last_updated_tick=tick,
        )
        self._store((source_kind, source_ref, effect_id, context_id), estimate)
        return estimate

    def rebuild(
        self,
        ledger: CausalEvidenceLedger,
        *,
        dimensions: Mapping[str, tuple[str, ...]] | None = None,
        view: OpportunityView | None = None,
    ) -> None:
        self._estimates = {}
        self._by_source = {}
        self._by_kind = {}
        keys = _source_keys(ledger)
        keys.update(_dimension_keys(ledger, dimensions or {}))
        for (kind, ref, effect_id, context_id), tick in sorted(
            keys.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3] or "")
        ):
            self.update_from_ledger(
                view if view is not None else ledger,
                source_kind=kind,
                source_ref=ref,
                effect_id=effect_id,
                context_id=context_id,
                tick=tick,
                dimension_signature_refs=(dimensions or {}).get(ref)
                if kind is CausalSourceKind.DIMENSION
                else None,
            )

    def estimate(
        self,
        *,
        source_kind: CausalSourceKind,
        source_ref: str,
        effect_id: str,
        context_id: str | None = None,
    ) -> ControllabilityEstimate | None:
        return self._estimates.get((source_kind, source_ref, effect_id, context_id))

    def discard_source(self, source_kind: CausalSourceKind, source_ref: str) -> None:
        for key in self._by_source.pop((source_kind, source_ref), set()):
            self._estimates.pop(key, None)
            self._by_kind.get(source_kind, set()).discard(key)


@dataclass(frozen=True, slots=True)
class AgencyEstimate:
    source_kind: CausalSourceKind
    source_ref: str

    effect_id: str
    context_id: str | None

    confidence: float
    temporal_contingency: float
    causal_specificity: float
    prediction_match: float | None
    support: int
    counterfactual_support: int
    last_updated_tick: int


class AgencyModel(_SourceIndexed):
    """Infer ownership of consequences without duplicating factual evidence.

    Agency is contingency + specificity + intervention evidence +
    counterfactual evidence + prediction + support (§23).  A matched
    prediction alone never yields agency: without positive temporal
    contingency *and* positive specificity against counterfactual windows the
    estimate is zero whatever the prediction agreement.
    """

    SCHEMA_VERSION = 1
    _estimate_type = AgencyEstimate

    @classmethod
    def restore(cls, payload: object) -> "AgencyModel":
        model = cls()
        model.restore_estimates(payload)
        return model


    def __init__(self) -> None:
        self._estimates: dict[_Key, AgencyEstimate] = {}
        self._by_source: dict[tuple[CausalSourceKind, str], set[_Key]] = {}
        self._by_kind: dict[CausalSourceKind, set[_Key]] = {}

    def update_from_ledger(
        self,
        ledger: OpportunityView,
        *,
        source_kind: CausalSourceKind,
        source_ref: str,
        effect_id: str,
        context_id: str | None,
        tick: int,
        prediction_match: float | None,
        dimension_signature_refs: tuple[str, ...] | None = None,
    ) -> AgencyEstimate:
        action_n, action_hits, other_n, other_hits = agency_source_opportunities(
            ledger,
            source_kind=source_kind,
            source_ref=source_ref,
            effect_id=effect_id,
            context_id=context_id,
            dimension_signature_refs=dimension_signature_refs,
        )
        temporal = action_hits / action_n if action_n else 0.0
        other_rate = other_hits / other_n if other_n else 0.0
        specificity = max(0.0, min(1.0, temporal - other_rate))
        confidence = agency_confidence(action_n, action_hits, other_n, other_hits, prediction_match)
        estimate = AgencyEstimate(
            source_kind=source_kind,
            source_ref=source_ref,
            effect_id=effect_id,
            context_id=context_id,
            confidence=confidence,
            temporal_contingency=temporal,
            causal_specificity=specificity,
            prediction_match=prediction_match,
            support=action_n,
            counterfactual_support=other_n,
            last_updated_tick=tick,
        )
        self._store((source_kind, source_ref, effect_id, context_id), estimate)
        return estimate

    def rebuild(
        self,
        ledger: CausalEvidenceLedger,
        *,
        dimensions: Mapping[str, tuple[str, ...]] | None = None,
        view: OpportunityView | None = None,
    ) -> None:
        self._estimates = {}
        self._by_source = {}
        self._by_kind = {}
        keys = _source_keys(ledger)
        keys.update(_dimension_keys(ledger, dimensions or {}))
        for (kind, ref, effect_id, context_id), tick in sorted(
            keys.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3] or "")
        ):
            self.update_from_ledger(
                view if view is not None else ledger,
                source_kind=kind,
                source_ref=ref,
                effect_id=effect_id,
                context_id=context_id,
                tick=tick,
                prediction_match=None,
                dimension_signature_refs=(dimensions or {}).get(ref)
                if kind is CausalSourceKind.DIMENSION
                else None,
            )

    def estimate(
        self,
        *,
        source_kind: CausalSourceKind,
        source_ref: str,
        effect_id: str,
        context_id: str | None = None,
    ) -> AgencyEstimate | None:
        return self._estimates.get((source_kind, source_ref, effect_id, context_id))

    def discard_source(self, source_kind: CausalSourceKind, source_ref: str) -> None:
        for key in self._by_source.pop((source_kind, source_ref), set()):
            self._estimates.pop(key, None)
            self._by_kind.get(source_kind, set()).discard(key)
