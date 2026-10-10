"""Causal Provenance v1 (docs/design/core/causal-provenance-v1.md).

One transversal contract for "why does this exist?": every state-changing
decision emits a ``CausalEvent`` naming what caused it and what it produced,
by stable, correlatable ``CausalRef``.  Timestamps say when; provenance says
why.

The organism keeps only a bounded ring of recent events and a **frontier**:
for every live reference, the refs of its immediate causes.  The frontier is
bounded by the number of live references, not by age, and is checkpointed, so
any live reference stays explainable after the ring wraps.  Durable history is
the lab's append-only journal, fed passively from the observation stream; it
never feeds back into the organism.

No raw values, no semantics: refs are structural kinds and opaque ids.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

RING_CAPACITY = 8192
MAX_CAUSES = 64
_EVENT_ID_PATTERN = re.compile(r"event\.[0-9a-f]{24}\Z")


@dataclass(frozen=True, slots=True, order=True)
class CausalRef:
    kind: str
    id: str

    def __post_init__(self) -> None:
        if not self.kind or not self.id:
            raise ValueError("a causal ref needs a kind and an id")

    def payload(self) -> list[str]:
        return [self.kind, self.id]

    @classmethod
    def from_payload(cls, raw: Iterable[str]) -> "CausalRef":
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or any(not isinstance(value, str) for value in raw)
        ):
            raise ValueError("causal reference must contain exactly two strings")
        kind, ref_id = raw
        return cls(kind, ref_id)


def _scalar(value: Any) -> float | int | str | bool | None:
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 12)
    raise ValueError("provenance parameters must be scalars")


@dataclass(frozen=True, slots=True)
class CausalEvent:
    tick: int
    domain: str
    operation: str
    subject: CausalRef
    caused_by: tuple[CausalRef, ...]
    produced: tuple[CausalRef, ...] = ()
    rule: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def __post_init__(self) -> None:
        if len(self.caused_by) > MAX_CAUSES:
            raise ValueError("a causal event names a bounded set of causes")
        params = {str(k): _scalar(v) for k, v in sorted(dict(self.parameters).items())}
        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "caused_by", tuple(self.caused_by))
        object.__setattr__(self, "produced", tuple(self.produced))
        digest = hashlib.sha256(
            json.dumps(self._identity(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        object.__setattr__(self, "event_id", "event." + digest)

    def _identity(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "domain": self.domain,
            "operation": self.operation,
            "subject": self.subject.payload(),
            "caused_by": [ref.payload() for ref in self.caused_by],
            "produced": [ref.payload() for ref in self.produced],
            "rule": self.rule,
            "parameters": dict(self.parameters),
        }

    def payload(self) -> dict[str, Any]:
        return {"event_id": self.event_id, **self._identity()}

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "CausalEvent":
        required = {
            "event_id",
            "tick",
            "domain",
            "operation",
            "subject",
            "caused_by",
            "produced",
            "rule",
            "parameters",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("invalid causal event payload fields")

        tick = raw["tick"]
        if isinstance(tick, bool) or not isinstance(tick, int):
            raise ValueError("causal event tick must be an integer")
        if not isinstance(raw["domain"], str) or not isinstance(raw["operation"], str):
            raise ValueError("causal event domain and operation must be strings")
        if not isinstance(raw["caused_by"], list) or not isinstance(raw["produced"], list):
            raise ValueError("causal event references must be lists")
        if not isinstance(raw["subject"], list) or any(
            not isinstance(ref, list)
            for ref in (*raw["caused_by"], *raw["produced"])
        ):
            raise ValueError("causal event references must use serialized lists")
        if raw["rule"] is not None and not isinstance(raw["rule"], str):
            raise ValueError("causal event rule must be a string or null")
        parameters = raw["parameters"]
        if not isinstance(parameters, Mapping) or any(
            not isinstance(key, str) for key in parameters
        ):
            raise ValueError("causal event parameters must be a string-keyed object")
        event_id = raw["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("causal event id must be a non-empty string")

        event = cls(
            tick=tick,
            domain=raw["domain"],
            operation=raw["operation"],
            subject=CausalRef.from_payload(raw["subject"]),
            caused_by=tuple(CausalRef.from_payload(item) for item in raw["caused_by"]),
            produced=tuple(CausalRef.from_payload(item) for item in raw["produced"]),
            rule=raw["rule"],
            parameters=parameters,
        )
        if event_id != event.event_id:
            raise ValueError("causal event id does not match its content")
        return event


class ProvenanceLog:
    """Bounded ring of recent events plus the frontier of live references."""

    SCHEMA_VERSION = 1

    def __init__(self, *, capacity: int = RING_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._ring: deque[CausalEvent] = deque(maxlen=self.capacity)
        self._frontier: dict[CausalRef, tuple[str, tuple[CausalRef, ...]]] = {}
        self._subscribers: list[Callable[[CausalEvent], None]] = []
        self.emitted = 0

    # -- emission -------------------------------------------------------------
    def subscribe(self, callback: Callable[[CausalEvent], None]) -> None:
        """Outward-only observers (e.g. the lab journal); never read back."""
        self._subscribers.append(callback)

    def emit(self, event: CausalEvent) -> CausalEvent:
        self._ring.append(event)
        self.emitted += 1
        for ref in event.produced or (event.subject,):
            self._frontier[ref] = (event.event_id, event.caused_by)
        for callback in self._subscribers:
            callback(event)
        return event

    def retire(self, refs: Iterable[CausalRef]) -> None:
        """A reference is no longer live: its frontier entry may go."""
        for ref in refs:
            self._frontier.pop(ref, None)

    # -- queries --------------------------------------------------------------
    def events(self) -> tuple[CausalEvent, ...]:
        return tuple(self._ring)

    def causes_of(self, ref: CausalRef) -> tuple[CausalRef, ...] | None:
        entry = self._frontier.get(ref)
        return entry[1] if entry is not None else None

    def cause_event(self, ref: CausalRef) -> str | None:
        entry = self._frontier.get(ref)
        return entry[0] if entry is not None else None

    def live_refs(self) -> tuple[CausalRef, ...]:
        return tuple(sorted(self._frontier))

    # -- persistence ----------------------------------------------------------
    def checkpoint(self) -> dict[str, Any]:
        """The frontier only: bounded by live references, never by age."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "emitted": self.emitted,
            "frontier": [
                {
                    "ref": ref.payload(),
                    "event_id": event_id,
                    "caused_by": [cause.payload() for cause in causes],
                }
                for ref, (event_id, causes) in sorted(self._frontier.items())
            ],
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "ProvenanceLog":
        required = {"schema_version", "capacity", "emitted", "frontier"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("invalid provenance checkpoint fields")
        version = payload["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != cls.SCHEMA_VERSION:
            raise ValueError("unsupported provenance checkpoint")
        capacity = payload["capacity"]
        emitted = payload["emitted"]
        frontier = payload["frontier"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("invalid provenance checkpoint capacity")
        if isinstance(emitted, bool) or not isinstance(emitted, int) or emitted < 0:
            raise ValueError("invalid provenance checkpoint emitted count")
        if not isinstance(frontier, list):
            raise ValueError("invalid provenance checkpoint frontier")
        log = cls(capacity=capacity)
        log.emitted = emitted
        for raw in frontier:
            if not isinstance(raw, Mapping) or set(raw) != {"ref", "event_id", "caused_by"}:
                raise ValueError("invalid provenance checkpoint frontier entry")
            ref = CausalRef.from_payload(raw["ref"])
            event_id = raw["event_id"]
            causes = raw["caused_by"]
            if (
                not isinstance(event_id, str)
                or _EVENT_ID_PATTERN.fullmatch(event_id) is None
                or not isinstance(causes, list)
                or len(causes) > MAX_CAUSES
            ):
                raise ValueError("invalid provenance checkpoint frontier entry")
            if ref in log._frontier:
                raise ValueError("duplicate provenance checkpoint reference")
            log._frontier[ref] = (
                event_id,
                tuple(CausalRef.from_payload(item) for item in causes),
            )
        return log


__all__ = ["CausalEvent", "CausalRef", "MAX_CAUSES", "ProvenanceLog", "RING_CAPACITY"]
