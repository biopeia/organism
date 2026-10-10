"""Capacity pressure of bounded stores (Cross-Domain Revision Coherence v1 §2.1).

Measurement only. A store notes an eviction *after* its existing policy has
chosen and applied it, and notes an admission after inserting a new item.
Nothing here is read by any decision, so instrumenting a store never changes
behaviour. Evicted items are remembered by a short one-way fingerprint so a
later reappearance counts as ``relearned_after_eviction``; the fingerprints
are never used as functional memory.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Iterable, Mapping

RECENT_EVICTIONS = 256


def _fingerprint(item_ref: str) -> str:
    return hashlib.sha256(item_ref.encode("utf-8")).hexdigest()[:16]


class CapacityPressure:
    SCHEMA_VERSION = 1

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.evictions = 0
        self.admissions = 0
        self.promotions = 0
        self.demotions = 0
        self.relearned_after_eviction = 0
        self._recently_evicted: OrderedDict[str, None] = OrderedDict()

    def note_evicted(self, item_refs: Iterable[str]) -> None:
        for item_ref in item_refs:
            self.evictions += 1
            key = _fingerprint(str(item_ref))
            self._recently_evicted[key] = None
            self._recently_evicted.move_to_end(key)
            while len(self._recently_evicted) > RECENT_EVICTIONS:
                self._recently_evicted.popitem(last=False)

    def note_admitted(self, item_ref: str) -> None:
        self.admissions += 1
        key = _fingerprint(str(item_ref))
        if key in self._recently_evicted:
            del self._recently_evicted[key]
            self.relearned_after_eviction += 1

    def snapshot(self, occupancy: int) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "occupancy": int(occupancy),
            "pressure": int(occupancy) / self.capacity,
            "admissions": self.admissions,
            "evictions": self.evictions,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "relearned_after_eviction": self.relearned_after_eviction,
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "admissions": self.admissions,
            "evictions": self.evictions,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "relearned_after_eviction": self.relearned_after_eviction,
            "recently_evicted": list(self._recently_evicted),
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any], *, capacity: int) -> "CapacityPressure":
        """Restore a complete TARGET checkpoint; fresh state uses the constructor."""
        required = {
            "schema_version",
            "capacity",
            "admissions",
            "evictions",
            "promotions",
            "demotions",
            "relearned_after_eviction",
            "recently_evicted",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("invalid capacity-pressure checkpoint fields")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != cls.SCHEMA_VERSION
        ):
            raise ValueError("unsupported capacity-pressure checkpoint")
        stored_capacity = payload["capacity"]
        if (
            isinstance(stored_capacity, bool)
            or not isinstance(stored_capacity, int)
            or stored_capacity != capacity
        ):
            raise ValueError("capacity-pressure capacity mismatch")
        counters = (
            "admissions",
            "evictions",
            "promotions",
            "demotions",
            "relearned_after_eviction",
        )
        for name in counters:
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid capacity-pressure {name}")
        raw_evicted = payload["recently_evicted"]
        if (
            not isinstance(raw_evicted, list)
            or len(raw_evicted) > RECENT_EVICTIONS
            or any(
                not isinstance(key, str)
                or len(key) != 16
                or any(char not in "0123456789abcdef" for char in key)
                for key in raw_evicted
            )
            or len(set(raw_evicted)) != len(raw_evicted)
        ):
            raise ValueError("invalid capacity-pressure recent evictions")

        pressure = cls(capacity)
        for name in counters:
            setattr(pressure, name, payload[name])
        for key in raw_evicted:
            pressure._recently_evicted[key] = None
        return pressure


__all__ = ["CapacityPressure", "RECENT_EVICTIONS"]
