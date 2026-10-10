"""Executive Outcome Learning v1.1 (docs/design/core/executive-outcome-learning-v1.md).

Real intent outcomes become local executive evidence that modulates future
*admission* of the same (competence, anticipated effect).  v1 also keyed on the
context; that fragmented memory so that almost no admission met its own
history (spec §12-§14), so v1.1 keys on the relation only.  This is executive
experience ("when I recently pursued C->E, did it work?"), kept apart from
physical causal belief ("can C cause E?"):
nothing here writes causal evidence, controllability, agency or competence
evidence, and nothing here is a reward.

Evidence stores facts (samples); the modulation policy is computed at lookup
so it can change without reinterpreting checkpoints.
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from statistics import fmean
from typing import Any, Mapping

from ..capacity import CapacityPressure

WINDOW = 16
MAX_KEYS = 256

ExecutiveKey = tuple[str, str]  # (competence_id, anticipated_effect_id)


class OutcomeClass(StrEnum):
    POSITIVE = "positive"
    CONTRADICTING = "contradicting"
    TERMINAL = "terminal"
    SUPPRESS = "suppress"
    NEUTRAL = "neutral"


# Reason strings mirror core.domains.intention (agency never imports core).
_POSITIVE = frozenset({"anticipated_effect_observed"})
_CONTRADICTING = frozenset(
    {
        "repeated_high_mismatch",
        "competence_exhausted_without_anticipated_consequence",
        "stagnation",
    }
)
_TERMINAL = frozenset({"controller_terminal_failure", "commitment_terminal_failure"})
_SUPPRESS = frozenset(
    {
        "surface_incompatible",
        "competence_no_longer_executable",
        "required_causal_binding_invalidated",
        "embodiment_changed",
    }
)


def classify_outcome(
    *, status: str, reason: str | None, binding_valid_at_start: bool
) -> OutcomeClass:
    """Map one terminal lifecycle outcome to what it says about C->E (spec §5)."""
    if status == "satisfied" and reason in _POSITIVE:
        return OutcomeClass.POSITIVE
    if status == "failed" and reason in _CONTRADICTING:
        return OutcomeClass.CONTRADICTING
    if status == "failed" and reason in _TERMINAL:
        # A controller that fails on infrastructure that was already invalid
        # says nothing about the competence.
        return OutcomeClass.TERMINAL if binding_valid_at_start else OutcomeClass.NEUTRAL
    if status == "invalidated" and reason in _SUPPRESS:
        return OutcomeClass.SUPPRESS
    return OutcomeClass.NEUTRAL


@dataclass(frozen=True, slots=True)
class ExecutiveOutcomeSample:
    tick: int
    outcome_class: OutcomeClass
    reason: str | None
    effect_similarity: float | None
    prediction_mismatch: float | None
    progress_before_failure: float | None


@dataclass(frozen=True, slots=True)
class SuppressionEvidence:
    """The causal/binding state an invalidation was judged against (spec §7)."""

    invalidated_tick: int
    reason: str
    binding_fingerprint: str | None
    executable: bool
    controllability_revision: int | None
    agency_revision: int | None

    def lifted_by(self, current: "CausalRevisionState") -> bool:
        return (
            current.binding_fingerprint != self.binding_fingerprint
            or current.executable != self.executable
            or current.controllability_revision != self.controllability_revision
            or current.agency_revision != self.agency_revision
        )


@dataclass(frozen=True, slots=True)
class CausalRevisionState:
    """Material causal/binding state of one relation, supplied by the action domain."""

    binding_fingerprint: str | None
    executable: bool
    controllability_revision: int | None
    agency_revision: int | None


class ExecutiveOutcomeEvidence:
    """Facts about one key: its last WINDOW counted outcomes and any suppression."""

    __slots__ = ("samples", "suppression")

    def __init__(self) -> None:
        self.samples: deque[ExecutiveOutcomeSample] = deque(maxlen=WINDOW)
        self.suppression: SuppressionEvidence | None = None

    def _count(self, outcome_class: OutcomeClass) -> int:
        return sum(1 for sample in self.samples if sample.outcome_class is outcome_class)

    @property
    def satisfactions(self) -> int:
        return self._count(OutcomeClass.POSITIVE)

    @property
    def contradicting_failures(self) -> int:
        return self._count(OutcomeClass.CONTRADICTING)

    @property
    def terminal_failures(self) -> int:
        return self._count(OutcomeClass.TERMINAL)

    @property
    def last_failure(self) -> ExecutiveOutcomeSample | None:
        return next(
            (
                sample
                for sample in reversed(self.samples)
                if sample.outcome_class in (OutcomeClass.CONTRADICTING, OutcomeClass.TERMINAL)
            ),
            None,
        )

    @property
    def last_failure_reason(self) -> str | None:
        failure = self.last_failure
        return failure.reason if failure is not None else None

    @property
    def progress_before_failure(self) -> float | None:
        failure = self.last_failure
        return failure.progress_before_failure if failure is not None else None

    @property
    def last_outcome_tick(self) -> int | None:
        return self.samples[-1].tick if self.samples else None

    @property
    def mean_prediction_mismatch(self) -> float | None:
        values = [
            sample.prediction_mismatch
            for sample in self.samples
            if sample.prediction_mismatch is not None
        ]
        return fmean(values) if values else None

    def checkpoint(self) -> dict[str, Any]:
        return {
            "samples": [
                {**asdict(sample), "outcome_class": sample.outcome_class.value}
                for sample in self.samples
            ],
            "suppression": asdict(self.suppression) if self.suppression is not None else None,
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "ExecutiveOutcomeEvidence":
        required = {"samples", "suppression"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("invalid executive outcome evidence fields")
        evidence = cls()
        raw_samples = payload["samples"]
        if not isinstance(raw_samples, list) or len(raw_samples) > WINDOW:
            raise ValueError("invalid or unbounded executive outcome samples")
        for raw in raw_samples:
            sample_fields = {
                "tick",
                "outcome_class",
                "reason",
                "effect_similarity",
                "prediction_mismatch",
                "progress_before_failure",
            }
            if not isinstance(raw, Mapping) or set(raw) != sample_fields:
                raise ValueError("invalid executive outcome sample fields")
            raw_class = raw["outcome_class"]
            if not isinstance(raw_class, str):
                raise ValueError("invalid executive outcome class")
            try:
                outcome_class = OutcomeClass(raw_class)
            except ValueError as exc:
                raise ValueError("invalid executive outcome class") from exc
            if outcome_class is OutcomeClass.NEUTRAL:
                raise ValueError("neutral outcomes are never executive evidence")
            tick = raw["tick"]
            reason = raw["reason"]
            if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
                raise ValueError("invalid executive outcome sample tick")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("invalid executive outcome sample reason")
            numeric_fields = (
                "effect_similarity",
                "prediction_mismatch",
                "progress_before_failure",
            )
            for name in numeric_fields:
                value = raw[name]
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"invalid executive outcome sample {name}")
            evidence.samples.append(
                ExecutiveOutcomeSample(
                    tick=tick,
                    outcome_class=outcome_class,
                    reason=reason,
                    effect_similarity=raw["effect_similarity"],
                    prediction_mismatch=raw["prediction_mismatch"],
                    progress_before_failure=raw["progress_before_failure"],
                )
            )
        raw_suppression = payload["suppression"]
        if raw_suppression is not None:
            suppression_fields = {
                "invalidated_tick",
                "reason",
                "binding_fingerprint",
                "executable",
                "controllability_revision",
                "agency_revision",
            }
            if not isinstance(raw_suppression, Mapping) or set(raw_suppression) != suppression_fields:
                raise ValueError("invalid executive outcome suppression fields")
            invalidated_tick = raw_suppression["invalidated_tick"]
            suppression_reason = raw_suppression["reason"]
            fingerprint = raw_suppression["binding_fingerprint"]
            executable = raw_suppression["executable"]
            if (
                isinstance(invalidated_tick, bool)
                or not isinstance(invalidated_tick, int)
                or invalidated_tick < 0
                or not isinstance(suppression_reason, str)
                or not suppression_reason
                or (fingerprint is not None and not isinstance(fingerprint, str))
                or not isinstance(executable, bool)
            ):
                raise ValueError("invalid executive outcome suppression")
            for name in ("controllability_revision", "agency_revision"):
                revision = raw_suppression[name]
                if revision is not None and (
                    isinstance(revision, bool)
                    or not isinstance(revision, int)
                    or revision < 0
                ):
                    raise ValueError("invalid executive outcome suppression revision")
            evidence.suppression = SuppressionEvidence(
                invalidated_tick=invalidated_tick,
                reason=suppression_reason,
                binding_fingerprint=fingerprint,
                executable=executable,
                controllability_revision=raw_suppression["controllability_revision"],
                agency_revision=raw_suppression["agency_revision"],
            )
        return evidence


@dataclass(frozen=True, slots=True)
class ExecutiveModulation:
    factor: float
    suppressed: bool
    had_history: bool


NO_HISTORY = ExecutiveModulation(factor=1.0, suppressed=False, had_history=False)


class ExecutiveAdmissionModulator:
    """Policy (spec §6): computed at lookup, never stored."""

    contradicting_weight: float = 1.0
    terminal_weight: float = 2.0
    min_factor: float = 0.5
    max_factor: float = 1.5

    def confidence(self, evidence: ExecutiveOutcomeEvidence) -> float:
        s = evidence.satisfactions
        c = evidence.contradicting_failures
        t = evidence.terminal_failures
        return (1.0 + s) / (2.0 + s + self.contradicting_weight * c + self.terminal_weight * t)

    def factor(self, evidence: ExecutiveOutcomeEvidence) -> float:
        return max(self.min_factor, min(self.max_factor, self.confidence(evidence) / 0.5))


def _key_ref(key: ExecutiveKey) -> list[str]:
    return [key[0], key[1]]


def _merge_into(target: ExecutiveOutcomeEvidence, source: ExecutiveOutcomeEvidence) -> None:
    """v1 -> v1.1: fold context-specific evidence into its (competence, effect) key."""
    merged = sorted((*target.samples, *source.samples), key=lambda sample: sample.tick)
    target.samples.clear()
    target.samples.extend(merged[-WINDOW:])
    candidates = [item for item in (target.suppression, source.suppression) if item is not None]
    target.suppression = max(candidates, key=lambda item: item.invalidated_tick, default=None)


class ExecutiveOutcomeLedger:
    """Bounded executive memory keyed by (competence, anticipated effect)."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        max_keys: int = MAX_KEYS,
        modulator: ExecutiveAdmissionModulator | None = None,
    ) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be positive")
        self.max_keys = int(max_keys)
        self.modulator = modulator or ExecutiveAdmissionModulator()
        self._items: OrderedDict[ExecutiveKey, ExecutiveOutcomeEvidence] = OrderedDict()
        self.keys_created = 0
        self.keys_evicted = 0
        self.lookups = 0
        self.history_hits = 0
        # Wave 0 measurement only (Cross-Domain Revision Coherence v1 §2.1).
        self.pressure = CapacityPressure(self.max_keys)

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: ExecutiveKey) -> ExecutiveOutcomeEvidence | None:
        return self._items.get(key)

    def _evidence_for(self, key: ExecutiveKey) -> ExecutiveOutcomeEvidence:
        evidence = self._items.get(key)
        if evidence is None:
            evidence = ExecutiveOutcomeEvidence()
            self._items[key] = evidence
            self.keys_created += 1
            self.pressure.note_admitted("|".join(key))
        self._items.move_to_end(key)  # least-recently-updated is evicted first
        while len(self._items) > self.max_keys:
            evicted, _ = self._items.popitem(last=False)
            self.keys_evicted += 1
            self.pressure.note_evicted(("|".join(evicted),))
        return evidence

    def suppressed_competences(self) -> frozenset[str]:
        """Competences with a live suppression record, read without side
        effects (no lifting, no lookup counting)."""
        return frozenset(
            key[0] for key, evidence in self._items.items() if evidence.suppression is not None
        )

    def record(
        self,
        key: ExecutiveKey,
        sample: ExecutiveOutcomeSample,
        *,
        revision: CausalRevisionState | None = None,
    ) -> None:
        """Record one real, non-neutral outcome for ``key``."""
        if sample.outcome_class is OutcomeClass.NEUTRAL:
            raise ValueError("neutral outcomes are never executive evidence")
        evidence = self._evidence_for(key)
        evidence.samples.append(sample)
        if sample.outcome_class is OutcomeClass.SUPPRESS:
            if revision is None:
                raise ValueError(
                    "suppression requires the causal revision state it is judged against"
                )
            evidence.suppression = SuppressionEvidence(
                invalidated_tick=sample.tick,
                reason=str(sample.reason),
                binding_fingerprint=revision.binding_fingerprint,
                executable=revision.executable,
                controllability_revision=revision.controllability_revision,
                agency_revision=revision.agency_revision,
            )

    def modulation(
        self, key: ExecutiveKey, *, revision: CausalRevisionState | None
    ) -> ExecutiveModulation:
        """Admission modulation for one candidate; lifts suppression on relevant revision."""
        self.lookups += 1
        evidence = self._items.get(key)
        if evidence is None:
            return NO_HISTORY
        self.history_hits += 1
        if evidence.suppression is not None:
            if revision is None or not evidence.suppression.lifted_by(revision):
                return ExecutiveModulation(factor=0.0, suppressed=True, had_history=True)
            evidence.suppression = None
        return ExecutiveModulation(
            factor=self.modulator.factor(evidence), suppressed=False, had_history=True
        )

    def metrics(self) -> dict[str, float | int | None]:
        sizes = [len(item.samples) for item in self._items.values()]
        return {
            "keys_created": self.keys_created,
            "keys_evicted": self.keys_evicted,
            "live_keys": len(self._items),
            "single_sample_key_fraction": (
                sum(1 for size in sizes if size == 1) / len(sizes) if sizes else None
            ),
            "mean_samples_per_key": fmean(sizes) if sizes else None,
            "lookups": self.lookups,
            "history_hits": self.history_hits,
            "history_hit_rate": self.history_hits / self.lookups if self.lookups else None,
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "entries": [
                {"key": _key_ref(key), **evidence.checkpoint()}
                for key, evidence in self._items.items()
            ],
            "counters": {
                "keys_created": self.keys_created,
                "keys_evicted": self.keys_evicted,
                "lookups": self.lookups,
                "history_hits": self.history_hits,
            },
            "capacity_pressure": self.pressure.checkpoint(),
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "ExecutiveOutcomeLedger":
        if not isinstance(payload, Mapping):
            raise ValueError("invalid executive outcome ledger checkpoint fields")
        version = payload.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int) or version != cls.SCHEMA_VERSION:
            raise ValueError("unsupported executive outcome ledger checkpoint")
        required = {"schema_version", "entries", "counters", "capacity_pressure"}
        if set(payload) != required:
            raise ValueError("invalid executive outcome ledger checkpoint fields")
        ledger = cls()
        entries = payload["entries"]
        if not isinstance(entries, list) or len(entries) > ledger.max_keys:
            raise ValueError("invalid or unbounded executive outcome ledger")
        for raw in entries:
            if not isinstance(raw, Mapping) or set(raw) != {"key", "samples", "suppression"}:
                raise ValueError("invalid executive outcome entry fields")
            raw_key = raw["key"]
            if (
                not isinstance(raw_key, list)
                or len(raw_key) != 2
                or any(not isinstance(part, str) or not part for part in raw_key)
            ):
                raise ValueError("invalid executive outcome key")
            key = (raw_key[0], raw_key[1])
            if key in ledger._items:
                raise ValueError("duplicate executive outcome key")
            evidence = ExecutiveOutcomeEvidence.restore(
                {name: raw[name] for name in ("samples", "suppression")}
            )
            ledger._items[key] = evidence
        counters = payload["counters"]
        if not isinstance(counters, Mapping) or set(counters) != {
            "keys_created",
            "keys_evicted",
            "lookups",
            "history_hits",
        }:
            raise ValueError("invalid executive outcome counters")
        for name in ("keys_created", "keys_evicted", "lookups", "history_hits"):
            value = counters[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("invalid executive outcome counter")
            setattr(ledger, name, value)
        ledger.pressure = CapacityPressure.restore(
            payload.get("capacity_pressure"), capacity=ledger.max_keys
        )
        return ledger


__all__ = [
    "CausalRevisionState",
    "ExecutiveAdmissionModulator",
    "ExecutiveKey",
    "ExecutiveModulation",
    "ExecutiveOutcomeEvidence",
    "ExecutiveOutcomeLedger",
    "ExecutiveOutcomeSample",
    "NO_HISTORY",
    "OutcomeClass",
    "SuppressionEvidence",
    "classify_outcome",
]
