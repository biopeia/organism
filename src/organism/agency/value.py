"""Outcome-value ledger for L8 Prospective Agency.

Maps opaque outcome tokens to endogenous historical value derived exclusively
from homeostatic deviation changes.  Value is learned at two resolutions:

* a global fallback for the opaque outcome token; and
* a context-specific estimate keyed by a digest of the organism's own private
  context tokens.

No environmental, semantic, evaluator-supplied, or physiology field names are
accepted here.  The context digest exists only to preserve the distinction
between otherwise identical outcomes experienced in different organism-visible
states.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

from .types import OutcomeValueEstimate

MAX_OUTCOME_VALUES = 256
MAX_CONTEXTUAL_OUTCOME_VALUES = 1024

# Confidence saturates at this many samples.
_CONFIDENCE_SATURATION_SAMPLES = 16
_CONTEXT_KEY_PREFIX = "value-context."


def context_key_for(context_tokens: Iterable[str]) -> str:
    """Return a stable opaque identity for an organism-visible context.

    Tokens are already private organism vocabulary.  The ledger deliberately
    stores only a digest so it cannot acquire or expose semantic interpretation
    of that vocabulary.  Order is preserved because the private model consumes
    the same ordered context sequence.
    """
    bounded: list[str] = []
    for raw in context_tokens:
        token = str(raw)
        if not token:
            continue
        bounded.append(token)
        if len(bounded) >= 512:
            break
    material = "\x1f".join(bounded).encode("utf-8")
    return _CONTEXT_KEY_PREFIX + hashlib.sha256(material).hexdigest()[:24]


def _valid_context_key(value: object) -> bool:
    return isinstance(value, str) and value.startswith(_CONTEXT_KEY_PREFIX) and len(value) <= 64


@dataclass(slots=True)
class OutcomeValueStat:
    """Online Welford statistics for one outcome token's historical value."""

    samples: int
    mean: float
    m2: float
    positive: int
    negative: int
    last_tick: int

    def update(self, value: float) -> None:
        self.samples += 1
        delta = value - self.mean
        self.mean += delta / self.samples
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if value > 0.0:
            self.positive += 1
        elif value < 0.0:
            self.negative += 1

    @property
    def variance(self) -> float:
        if self.samples < 2:
            return 0.0
        return self.m2 / (self.samples - 1)

    def to_estimate(self, outcome_id: str) -> OutcomeValueEstimate:
        confidence = min(1.0, self.samples / float(_CONFIDENCE_SATURATION_SAMPLES))
        return OutcomeValueEstimate(
            outcome_id=outcome_id,
            samples=self.samples,
            mean_value=self.mean,
            variance=self.variance,
            confidence=max(1e-9, confidence),
        )

    def checkpoint(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "mean": self.mean,
            "m2": self.m2,
            "positive": self.positive,
            "negative": self.negative,
            "last_tick": self.last_tick,
        }

    @classmethod
    def restore(cls, payload: dict[str, Any]) -> "OutcomeValueStat":
        def _int(v: object, name: str, minimum: int = 0) -> int:
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError(f"invalid {name} in outcome-value stat: {v!r}")
            return int(v)

        def _float(v: object, name: str) -> float:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"invalid {name} in outcome-value stat")
            if not math.isfinite(float(v)):
                raise ValueError(f"{name} must be finite in outcome-value stat")
            return float(v)

        return cls(
            samples=_int(payload.get("samples", 0), "samples", minimum=1),
            mean=_float(payload.get("mean", 0.0), "mean"),
            m2=_float(payload.get("m2", 0.0), "m2"),
            positive=_int(payload.get("positive", 0), "positive"),
            negative=_int(payload.get("negative", 0), "negative"),
            last_tick=_int(payload.get("last_tick", 0), "last_tick"),
        )


class OutcomeValueLedger:
    """Learn endogenous value of opaque outcomes without erasing context.

    Every observation updates the global fallback.  When ``context_key`` is
    supplied it also updates a context-specific estimate.  Prospective lookup
    prefers the contextual estimate and falls back to global evidence while a
    new context is still evidence-poor.
    """

    _SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self._stats: dict[str, OutcomeValueStat] = {}
        self._context_stats: dict[tuple[str, str], OutcomeValueStat] = {}

    @property
    def known_outcome_count(self) -> int:
        return len(self._stats)

    @property
    def contextual_outcome_count(self) -> int:
        return len(self._context_stats)

    @staticmethod
    def _new_stat(tick: int) -> OutcomeValueStat:
        return OutcomeValueStat(
            samples=0,
            mean=0.0,
            m2=0.0,
            positive=0,
            negative=0,
            last_tick=tick,
        )

    @staticmethod
    def _update_stat(stat: OutcomeValueStat, value: float, *, tick: int) -> None:
        stat.update(value)
        stat.last_tick = tick

    def observe(
        self,
        outcome_id: str,
        intrinsic_value: float,
        *,
        tick: int,
        context_key: str | None = None,
    ) -> None:
        """Record one factual endogenous-value observation.

        ``context_key`` must be an opaque key generated from organism-visible
        context.  Supplying no context preserves the v1 global-only behaviour.
        """
        if not isinstance(outcome_id, str) or not outcome_id or len(outcome_id) > 128:
            raise ValueError("outcome_id must be a bounded non-empty string")
        if context_key is not None and not _valid_context_key(context_key):
            raise ValueError("context_key must be an opaque value-context key")
        if (
            isinstance(intrinsic_value, bool)
            or not isinstance(intrinsic_value, (int, float))
            or not math.isfinite(float(intrinsic_value))
        ):
            return
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a non-negative integer")

        value = max(-1.0, min(1.0, float(intrinsic_value)))

        stat = self._stats.get(outcome_id)
        if stat is None:
            if len(self._stats) >= MAX_OUTCOME_VALUES:
                self._evict_global_one()
            stat = self._new_stat(tick)
            self._stats[outcome_id] = stat
        self._update_stat(stat, value, tick=tick)

        if context_key is None:
            return
        contextual_key = (context_key, outcome_id)
        contextual = self._context_stats.get(contextual_key)
        if contextual is None:
            if len(self._context_stats) >= MAX_CONTEXTUAL_OUTCOME_VALUES:
                self._evict_contextual_one()
            contextual = self._new_stat(tick)
            self._context_stats[contextual_key] = contextual
        self._update_stat(contextual, value, tick=tick)

    def _evict_global_one(self) -> None:
        if not self._stats:
            return
        worst_key = min(
            self._stats,
            key=lambda k: (self._stats[k].samples, self._stats[k].last_tick, k),
        )
        del self._stats[worst_key]

    def _evict_contextual_one(self) -> None:
        if not self._context_stats:
            return
        worst_key = min(
            self._context_stats,
            key=lambda k: (
                self._context_stats[k].samples,
                self._context_stats[k].last_tick,
                k,
            ),
        )
        del self._context_stats[worst_key]

    def estimate(
        self,
        outcome_id: str,
        *,
        context_key: str | None = None,
    ) -> OutcomeValueEstimate | None:
        """Return contextual value when known, otherwise the global fallback."""
        if context_key is not None:
            if not _valid_context_key(context_key):
                raise ValueError("context_key must be an opaque value-context key")
            contextual = self._context_stats.get((context_key, outcome_id))
            if contextual is not None and contextual.samples >= 1:
                return contextual.to_estimate(outcome_id)
        stat = self._stats.get(outcome_id)
        if stat is None or stat.samples < 1:
            return None
        return stat.to_estimate(outcome_id)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self._SCHEMA_VERSION,
            "stats": {outcome_id: stat.checkpoint() for outcome_id, stat in self._stats.items()},
            "context_stats": [
                {
                    "context_key": context_key,
                    "outcome_id": outcome_id,
                    "stat": stat.checkpoint(),
                }
                for (context_key, outcome_id), stat in sorted(self._context_stats.items())
            ],
        }

    @classmethod
    def restore(cls, payload: object) -> "OutcomeValueLedger":
        """Restore v1 global ledgers or v2 contextual ledgers."""
        if not isinstance(payload, dict):
            raise ValueError("invalid outcome-value ledger checkpoint")
        version = payload.get("schema_version")
        if version not in {1, cls._SCHEMA_VERSION}:
            raise ValueError(f"unsupported outcome-value ledger schema version: {version!r}")
        raw_stats = payload.get("stats", {})
        if not isinstance(raw_stats, dict):
            raise ValueError("invalid outcome-value ledger stats checkpoint")

        ledger = cls()
        for outcome_id, raw_stat in raw_stats.items():
            if len(ledger._stats) >= MAX_OUTCOME_VALUES:
                break
            if not isinstance(outcome_id, str) or not isinstance(raw_stat, dict):
                continue
            try:
                ledger._stats[outcome_id] = OutcomeValueStat.restore(raw_stat)
            except (ValueError, TypeError, KeyError):
                continue

        if version == cls._SCHEMA_VERSION:
            raw_context = payload.get("context_stats", [])
            if not isinstance(raw_context, list):
                raise ValueError("invalid contextual outcome-value checkpoint")
            for item in raw_context:
                if len(ledger._context_stats) >= MAX_CONTEXTUAL_OUTCOME_VALUES:
                    break
                if not isinstance(item, dict):
                    continue
                context_key = item.get("context_key")
                outcome_id = item.get("outcome_id")
                raw_stat = item.get("stat")
                if (
                    not _valid_context_key(context_key)
                    or not isinstance(outcome_id, str)
                    or not outcome_id
                    or not isinstance(raw_stat, dict)
                ):
                    continue
                try:
                    ledger._context_stats[(context_key, outcome_id)] = OutcomeValueStat.restore(
                        raw_stat
                    )
                except (ValueError, TypeError, KeyError):
                    continue
        return ledger


__all__ = [
    "MAX_CONTEXTUAL_OUTCOME_VALUES",
    "MAX_OUTCOME_VALUES",
    "OutcomeValueStat",
    "OutcomeValueLedger",
    "context_key_for",
]
