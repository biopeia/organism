"""Embodiment-local execution authority for learned motor competences."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

from ..capacity import CapacityPressure
from .competence import CompetenceMaturity, MotorCompetence


class BindingStatus(StrEnum):
    """Cross-Domain Revision Coherence v1 §3.2."""

    VALID = "valid"
    STALE = "stale"
    INVALIDATED = "invalidated"


class StalenessReason(StrEnum):
    SURFACE_NOT_CURRENT = "surface_not_current"
    EMBODIMENT_NOT_CURRENT = "embodiment_not_current"
    CONTROLLER_UNAVAILABLE = "controller_unavailable"
    EVIDENCE_AGED = "evidence_aged"


class InvalidationReason(StrEnum):
    CAUSAL_RELATION_REVISED = "causal_relation_revised"
    EFFECT_SUPERSEDED = "effect_superseded"
    EVIDENCE_CONTRADICTED = "evidence_contradicted"


@dataclass(frozen=True, slots=True)
class CompetenceExecutionBinding:
    """Current-body evidence that one general competence can execute here.

    ``reliability`` is historical; current executability is ``status``.
    ``last_evidence_tick`` is the last confirmation, independent of
    ``status_changed_tick`` (an invalidation may follow the last confirmation).
    """

    competence_id: str
    surface_fingerprint: str
    effect_id: str
    evidence_refs: tuple[str, ...]
    reliability: float
    # Binding/execution metric supplied by owner contract; not a causal estimate.
    controllability: float
    last_evidence_tick: int
    status: BindingStatus = BindingStatus.VALID
    status_reason: StalenessReason | InvalidationReason | None = None
    valid_from_tick: int = 0
    status_changed_tick: int = 0
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.competence_id or not self.surface_fingerprint or not self.effect_id:
            raise ValueError("execution binding identifiers must be non-empty")
        if not self.evidence_refs:
            raise ValueError("execution binding requires current factual evidence")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("binding reliability must be in [0,1]")
        if not 0.0 <= self.controllability <= 1.0:
            raise ValueError("binding controllability must be in [0,1]")
        if self.last_evidence_tick < 0:
            raise ValueError("last_evidence_tick must be non-negative")
        if self.status is BindingStatus.VALID and self.status_reason is not None:
            raise ValueError("a valid binding carries no status reason")
        if self.status is BindingStatus.STALE and not isinstance(
            self.status_reason, StalenessReason
        ):
            raise ValueError("a stale binding needs a staleness reason")
        if self.status is BindingStatus.INVALIDATED and not isinstance(
            self.status_reason, InvalidationReason
        ):
            raise ValueError("an invalidated binding needs an invalidation reason")
        if min(self.valid_from_tick, self.status_changed_tick, self.revision) < 0:
            raise ValueError("binding lifecycle ticks and revision must be non-negative")


@dataclass(frozen=True, slots=True)
class BindingTransition:
    """One lifecycle change, for the caller to trace (previous state lives here only)."""

    competence_id: str
    previous_status: BindingStatus
    previous_revision: int
    binding: CompetenceExecutionBinding
    cause_refs: tuple[str, ...] = ()


class CompetenceExecutionBindingRegistry:
    """Bounded current-embodiment authority, separate from competence knowledge."""

    SCHEMA_VERSION = 3

    def __init__(self, *, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: dict[str, CompetenceExecutionBinding] = {}
        # Wave 0 measurement only (Cross-Domain Revision Coherence v1 §2.1).
        self.pressure = CapacityPressure(self.capacity)
        # Lifecycle changes not yet traced by the owner of provenance.
        self._transitions: list[BindingTransition] = []
        # Executions since each binding's last confirmation that did not
        # confirm it (binding evidence for §3.9 decision 1).
        self._unconfirmed: dict[str, dict[str, object]] = {}
        # Binding Degradation v1 §2: per-binding revision history (reference
        # and later executions); cleared when a new revision starts.
        self._history: dict[str, dict[str, Any]] = {}

    def bind_from_evidence(
        self,
        *,
        competence_id: str,
        surface_fingerprint: str,
        effect_id: str,
        evidence_refs: tuple[str, ...],
        reliability: float,
        controllability: float,
        tick: int,
    ) -> CompetenceExecutionBinding:
        existing = self._items.get(competence_id)
        refs = tuple(
            dict.fromkeys(
                (existing.evidence_refs if existing is not None else ()) + tuple(evidence_refs)
            )
        )
        revalidated = existing is not None and existing.status is not BindingStatus.VALID
        binding = CompetenceExecutionBinding(
            competence_id=competence_id,
            surface_fingerprint=surface_fingerprint,
            effect_id=effect_id,
            evidence_refs=refs,
            reliability=max(0.0, min(1.0, float(reliability))),
            controllability=max(0.0, min(1.0, float(controllability))),
            last_evidence_tick=int(tick),
            valid_from_tick=(
                existing.valid_from_tick if existing is not None and not revalidated else int(tick)
            ),
            status_changed_tick=(
                int(tick) if existing is None or revalidated else existing.status_changed_tick
            ),
            revision=(existing.revision + int(revalidated)) if existing is not None else 0,
        )
        self._items[competence_id] = binding
        self._unconfirmed.pop(competence_id, None)  # confirmed
        if revalidated:
            self._transitions.append(
                BindingTransition(competence_id, existing.status, existing.revision, binding)
            )
        if existing is None:
            self.pressure.note_admitted(competence_id)
        self._enforce_bound()
        return binding

    def _enforce_bound(self) -> None:
        if len(self._items) <= self.capacity:
            return
        retained = sorted(
            self._items.values(),
            key=lambda item: (
                -item.last_evidence_tick,
                -item.controllability,
                -item.reliability,
                item.competence_id,
            ),
        )[: self.capacity]
        before = tuple(self._items)
        self._items = {item.competence_id: item for item in retained}
        self.pressure.note_evicted(ref for ref in before if ref not in self._items)

    def get(self, competence_id: str) -> CompetenceExecutionBinding | None:
        return self._items.get(competence_id)

    def set_status(
        self,
        competence_id: str,
        status: BindingStatus,
        reason: StalenessReason | InvalidationReason | None,
        *,
        tick: int,
        cause_refs: tuple[str, ...] = (),
    ) -> BindingTransition | None:
        """Apply a lifecycle change decided by binding evidence; None if unchanged.

        Leaving INVALIDATED requires new evidence (``bind_from_evidence``).
        """
        binding = self._items.get(competence_id)
        if binding is None or (binding.status, binding.status_reason) == (status, reason):
            return None
        if binding.status is BindingStatus.INVALIDATED:
            return None
        changed = replace(
            binding,
            status=status,
            status_reason=reason,
            status_changed_tick=int(tick),
            valid_from_tick=int(tick) if status is BindingStatus.VALID else binding.valid_from_tick,
            revision=binding.revision + 1,
        )
        self._items[competence_id] = changed
        transition = BindingTransition(
            competence_id, binding.status, binding.revision, changed, tuple(cause_refs)
        )
        self._transitions.append(transition)
        return transition

    def note_unconfirmed_execution(
        self, competence_id: str, *, windows: int, commitment_id: str
    ) -> tuple[int, float, tuple[str, ...]]:
        """Record one execution that did not confirm the binding.

        Returns (executions, mean windows, recent commitment ids) since the
        binding's last confirmation.
        """
        entry = self._unconfirmed.setdefault(
            competence_id, {"executions": 0, "windows": 0, "commitments": []}
        )
        entry["executions"] = int(entry["executions"]) + 1
        entry["windows"] = int(entry["windows"]) + max(1, int(windows))
        commitments = [*list(entry["commitments"]), commitment_id][-8:]
        entry["commitments"] = commitments
        executions = int(entry["executions"])
        return executions, int(entry["windows"]) / executions, tuple(commitments)

    def note_execution(
        self, competence_id: str, *, matched: bool, commitment_id: str, reference_size: int
    ) -> dict[str, Any] | None:
        """Binding Degradation v1 §2: count one execution of a VALID binding.

        The first ``reference_size`` executions of the current revision form
        the frozen reference; later ones are degradation evidence. Each
        commitment counts once.
        """
        binding = self._items.get(competence_id)
        if binding is None or binding.status is not BindingStatus.VALID:
            return None
        entry = self._history.get(competence_id)
        if entry is None or entry["revision"] != binding.revision:
            entry = {
                "revision": binding.revision,
                "ref_n": 0,
                "ref_k": 0,
                "new_n": 0,
                "new_k": 0,
                "commitments": [],
            }
            self._history[competence_id] = entry
        counted = list(entry["commitments"])
        if commitment_id in counted:
            return entry
        if int(entry["ref_n"]) < reference_size:
            entry["ref_n"] = int(entry["ref_n"]) + 1
            entry["ref_k"] = int(entry["ref_k"]) + int(matched)
        else:
            entry["new_n"] = int(entry["new_n"]) + 1
            entry["new_k"] = int(entry["new_k"]) + int(matched)
        entry["commitments"] = [*counted, commitment_id][-64:]
        return entry

    def drain_transitions(self) -> tuple[BindingTransition, ...]:
        drained = tuple(self._transitions)
        self._transitions.clear()
        return drained

    def is_executable(
        self,
        competence: MotorCompetence,
        *,
        surface_fingerprint: str | None,
    ) -> bool:
        if surface_fingerprint is None:
            return False
        binding = self._items.get(competence.competence_id)
        return (
            binding is not None
            and binding.status is BindingStatus.VALID
            and binding.surface_fingerprint == surface_fingerprint
            and bool(binding.evidence_refs)
            and competence.maturity in {CompetenceMaturity.ESTABLISHED, CompetenceMaturity.ROBUST}
        )

    def invalid_for_surface(self, surface_fingerprint: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                competence_id
                for competence_id, binding in self._items.items()
                if binding.surface_fingerprint != surface_fingerprint
            )
        )

    @property
    def items(self) -> tuple[CompetenceExecutionBinding, ...]:
        return tuple(sorted(self._items.values(), key=lambda item: item.competence_id))

    def checkpoint(self) -> dict[str, object]:
        def binding_payload(item: CompetenceExecutionBinding) -> dict[str, object]:
            return {
                "competence_id": item.competence_id,
                "surface_fingerprint": item.surface_fingerprint,
                "effect_id": item.effect_id,
                "evidence_refs": list(item.evidence_refs),
                "reliability": item.reliability,
                "controllability": item.controllability,
                "last_evidence_tick": item.last_evidence_tick,
                "status": item.status.value,
                "status_reason": (
                    item.status_reason.value if item.status_reason is not None else None
                ),
                "valid_from_tick": item.valid_from_tick,
                "status_changed_tick": item.status_changed_tick,
                "revision": item.revision,
            }

        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "items": [binding_payload(item) for item in self.items],
            "pending_transitions": [
                {
                    "competence_id": transition.competence_id,
                    "previous_status": transition.previous_status.value,
                    "previous_revision": transition.previous_revision,
                    "binding": binding_payload(transition.binding),
                    "cause_refs": list(transition.cause_refs),
                }
                for transition in self._transitions
            ],
            "capacity_pressure": self.pressure.checkpoint(),
            "revision_history": {
                competence_id: dict(entry) for competence_id, entry in sorted(self._history.items())
            },
            "unconfirmed_executions": {
                competence_id: dict(entry)
                for competence_id, entry in sorted(self._unconfirmed.items())
            },
        }

    @classmethod
    def restore(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompetenceExecutionBindingRegistry":
        required = {
            "schema_version",
            "capacity",
            "items",
            "pending_transitions",
            "capacity_pressure",
            "revision_history",
            "unconfirmed_executions",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("invalid competence execution binding checkpoint fields")
        version = payload["schema_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != cls.SCHEMA_VERSION
        ):
            raise ValueError("unsupported competence execution binding checkpoint")
        capacity = payload["capacity"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("invalid competence execution binding capacity")
        pressure = payload["capacity_pressure"]
        if (
            not isinstance(pressure, Mapping)
            or pressure.get("capacity") != capacity
        ):
            raise ValueError("incomplete capacity pressure checkpoint")
        if not isinstance(payload["revision_history"], Mapping) or not isinstance(
            payload["unconfirmed_executions"], Mapping
        ):
            raise ValueError("invalid competence execution binding metadata")
        obj = cls(capacity=capacity)
        raw = payload["items"]
        if not isinstance(raw, list) or len(raw) > obj.capacity:
            raise ValueError("invalid or unbounded competence execution bindings")
        for entry in raw:
            binding = _restore_binding(entry)
            if binding.competence_id in obj._items:
                raise ValueError("duplicate competence execution binding identity")
            obj._items[binding.competence_id] = binding
        raw_transitions = payload["pending_transitions"]
        if not isinstance(raw_transitions, list):
            raise ValueError("invalid pending competence execution binding transitions")
        for entry in raw_transitions:
            if not isinstance(entry, Mapping) or set(entry) != {
                "competence_id",
                "previous_status",
                "previous_revision",
                "binding",
                "cause_refs",
            }:
                raise ValueError("invalid pending competence execution binding transition")
            raw_binding = entry["binding"]
            binding = _restore_binding(raw_binding)
            cause_refs = entry["cause_refs"]
            if not isinstance(cause_refs, list) or not all(
                isinstance(value, str) and value for value in cause_refs
            ):
                raise ValueError("invalid pending competence execution binding cause references")
            previous_revision = entry["previous_revision"]
            if (
                isinstance(previous_revision, bool)
                or not isinstance(previous_revision, int)
                or previous_revision < 0
            ):
                raise ValueError("invalid previous competence execution binding revision")
            if entry["competence_id"] != binding.competence_id:
                raise ValueError("pending competence execution binding identity mismatch")
            if binding.revision != previous_revision + 1:
                raise ValueError("invalid pending competence execution binding revision")
            previous_status = entry["previous_status"]
            if not isinstance(previous_status, str):
                raise ValueError("invalid previous competence execution binding status")
            obj._transitions.append(
                BindingTransition(
                    competence_id=binding.competence_id,
                    previous_status=BindingStatus(previous_status),
                    previous_revision=previous_revision,
                    binding=binding,
                    cause_refs=tuple(cause_refs),
                )
            )
        obj.pressure = CapacityPressure.restore(pressure, capacity=obj.capacity)
        for competence_id, entry in payload["revision_history"].items():
            if not isinstance(competence_id, str) or not competence_id:
                raise ValueError("invalid competence execution binding revision identity")
            fields = {"revision", "ref_n", "ref_k", "new_n", "new_k", "commitments"}
            if not isinstance(entry, Mapping) or set(entry) != fields:
                raise ValueError("invalid competence execution binding revision history")
            counts = {name: entry[name] for name in ("revision", "ref_n", "ref_k", "new_n", "new_k")}
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
                raise ValueError("invalid competence execution binding revision counts")
            commitments = entry["commitments"]
            if not _valid_string_list(commitments, limit=64):
                raise ValueError("invalid competence execution binding revision commitments")
            if counts["ref_k"] > counts["ref_n"] or counts["new_k"] > counts["new_n"]:
                raise ValueError("inconsistent competence execution binding revision history")
            obj._history[competence_id] = {
                **counts,
                "commitments": list(commitments),
            }
        for competence_id, entry in payload["unconfirmed_executions"].items():
            if not isinstance(competence_id, str) or not competence_id:
                raise ValueError("invalid unconfirmed competence execution identity")
            fields = {"executions", "windows", "commitments"}
            if not isinstance(entry, Mapping) or set(entry) != fields:
                raise ValueError("invalid unconfirmed competence execution metadata")
            executions, windows = entry["executions"], entry["windows"]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (executions, windows)):
                raise ValueError("invalid unconfirmed competence execution counts")
            commitments = entry["commitments"]
            # Each unconfirmed execution is counted independently. The writer
            # preserves repeated commitment IDs when the same commitment is
            # observed more than once, so restore must preserve those entries.
            if not _valid_string_list(commitments, limit=8, unique=False):
                raise ValueError("invalid unconfirmed competence execution commitments")
            obj._unconfirmed[competence_id] = {
                "executions": executions,
                "windows": windows,
                "commitments": list(commitments),
            }
        return obj


_BINDING_FIELDS = {
    "competence_id",
    "surface_fingerprint",
    "effect_id",
    "evidence_refs",
    "reliability",
    "controllability",
    "last_evidence_tick",
    "status",
    "status_reason",
    "valid_from_tick",
    "status_changed_tick",
    "revision",
}


def _restore_binding(raw: object) -> CompetenceExecutionBinding:
    if not isinstance(raw, Mapping) or set(raw) != _BINDING_FIELDS:
        raise ValueError("invalid competence execution binding fields")
    for name in ("competence_id", "surface_fingerprint", "effect_id"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise ValueError("invalid competence execution binding identity")
    refs = raw["evidence_refs"]
    if not _valid_string_list(refs):
        raise ValueError("invalid competence execution binding evidence references")
    for name in ("reliability", "controllability"):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid competence execution binding confidence")
    for name in ("last_evidence_tick", "valid_from_tick", "status_changed_tick", "revision"):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("invalid competence execution binding integer field")
    status = raw["status"]
    if not isinstance(status, str):
        raise ValueError("invalid competence execution binding status")
    reason = raw["status_reason"]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("invalid competence execution binding status reason")
    return CompetenceExecutionBinding(
        competence_id=raw["competence_id"],
        surface_fingerprint=raw["surface_fingerprint"],
        effect_id=raw["effect_id"],
        evidence_refs=tuple(refs),
        reliability=raw["reliability"],
        controllability=raw["controllability"],
        last_evidence_tick=raw["last_evidence_tick"],
        status=BindingStatus(status),
        status_reason=_reason(reason),
        valid_from_tick=raw["valid_from_tick"],
        status_changed_tick=raw["status_changed_tick"],
        revision=raw["revision"],
    )


def _valid_string_list(
    raw: object, *, limit: int | None = None, unique: bool = True
) -> bool:
    return (
        isinstance(raw, list)
        and (limit is None or len(raw) <= limit)
        and all(isinstance(value, str) and bool(value) for value in raw)
        and (not unique or len(set(raw)) == len(raw))
    )


def _reason(raw: object) -> StalenessReason | InvalidationReason | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("invalid competence execution binding status reason")
    value = raw
    if value in StalenessReason._value2member_map_:
        return StalenessReason(value)
    return InvalidationReason(value)


__all__ = [
    "BindingStatus",
    "BindingTransition",
    "CompetenceExecutionBinding",
    "CompetenceExecutionBindingRegistry",
    "InvalidationReason",
    "StalenessReason",
]
