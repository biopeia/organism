"""Organism-owned continuation identity and process-local canonical authority.

The registry coordinates repeated restores of the same serialized authority
lineage within a process, including stale-snapshot rejection after registry
objects are collected. Durable storage must still provide cross-process
compare-and-commit, and independent/copied authorities are not globally
coordinated.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
from threading import RLock
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

from .checkpoint import InvalidOrganismCheckpoint, OrganismCheckpoint
from .identity import OrganismIdentity
from .transition import OrganismLifecycle, OrganismTransitionState


class ContinuationError(ValueError):
    """Invalid continuation or canonical-authority operation."""


class DuplicateCanonicalContinuation(ContinuationError):
    """A canonical continuation is already active in this registry."""


class StaleCanonicalContinuation(ContinuationError):
    """A checkpoint is not the registry's current canonical state."""


class InvalidContinuationCheckpoint(ContinuationError):
    """A continuation envelope is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class ContinuationId:
    """Stable identity for one continuation, distinct from Organism identity."""

    value: UUID

    @classmethod
    def create(cls, value: UUID) -> "ContinuationId":
        if not isinstance(value, UUID):
            raise TypeError("continuation id must be an explicitly supplied UUID")
        return cls(value)

    @classmethod
    def parse(cls, value: str) -> "ContinuationId":
        return cls(UUID(value))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class ContinuationAncestry:
    """Provenance linking a new Organism identity to its source checkpoint."""

    parent_organism_id: OrganismIdentity
    parent_continuation_id: ContinuationId
    parent_organism_step: int
    source_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_organism_id, OrganismIdentity):
            raise ContinuationError("parent_organism_id must be an OrganismIdentity")
        if not isinstance(self.parent_continuation_id, ContinuationId):
            raise ContinuationError("parent_continuation_id must be a ContinuationId")
        if type(self.parent_organism_step) is not int or self.parent_organism_step < 0:
            raise ContinuationError("parent_organism_step must be non-negative")
        if (
            len(self.source_checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_checkpoint_sha256)
        ):
            raise ContinuationError("source checkpoint reference must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class OrganismContinuation:
    """Portable continuation metadata; it does not confer canonical authority."""

    state: OrganismTransitionState
    continuation_id: ContinuationId
    ancestry: ContinuationAncestry | None = None

    SCHEMA = "biopeia.organism-continuation"
    SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        if not isinstance(self.state, OrganismTransitionState):
            raise ContinuationError("state must be OrganismTransitionState")
        if not isinstance(self.continuation_id, ContinuationId):
            raise ContinuationError("continuation_id must be ContinuationId")
        if self.ancestry is not None and not isinstance(self.ancestry, ContinuationAncestry):
            raise ContinuationError("ancestry must be ContinuationAncestry or None")
        if self.ancestry is not None and self.ancestry.parent_organism_id == self.state.identity:
            raise ContinuationError("a branch must use a new Organism identity")

    @property
    def identity(self) -> OrganismIdentity:
        return self.state.identity

    @property
    def organism_step(self) -> int:
        return self.state.organism_step

    def to_bytes(self) -> bytes:
        ancestry = None
        if self.ancestry is not None:
            ancestry = {
                "parent_organism_id": str(self.ancestry.parent_organism_id),
                "parent_continuation_id": str(self.ancestry.parent_continuation_id),
                "parent_organism_step": self.ancestry.parent_organism_step,
                "source_checkpoint_sha256": self.ancestry.source_checkpoint_sha256,
            }
        document = {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "continuation_id": str(self.continuation_id),
            "owner_transition_base64": base64.b64encode(self.state.to_bytes()).decode("ascii"),
            "ancestry": ancestry,
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OrganismContinuation":
        expected = {"schema", "schema_version", "continuation_id", "owner_transition_base64", "ancestry"}
        try:
            doc = json.loads(payload)
            if not isinstance(doc, dict) or set(doc) != expected:
                raise InvalidContinuationCheckpoint("continuation checkpoint has an invalid shape")
            if doc["schema"] != cls.SCHEMA or type(doc["schema_version"]) is not int or doc["schema_version"] != cls.SCHEMA_VERSION:
                raise InvalidContinuationCheckpoint("unsupported continuation checkpoint schema")
            state_bytes = base64.b64decode(doc["owner_transition_base64"], validate=True)
            state = OrganismTransitionState.from_bytes(state_bytes)
            raw = doc["ancestry"]
            ancestry = None
            if raw is not None:
                if not isinstance(raw, dict) or set(raw) != {
                    "parent_organism_id", "parent_continuation_id", "parent_organism_step", "source_checkpoint_sha256"
                }:
                    raise InvalidContinuationCheckpoint("continuation ancestry has an invalid shape")
                ancestry = ContinuationAncestry(
                    OrganismIdentity.parse(raw["parent_organism_id"]),
                    ContinuationId.parse(raw["parent_continuation_id"]),
                    raw["parent_organism_step"],
                    raw["source_checkpoint_sha256"],
                )
            return cls(state, ContinuationId.parse(doc["continuation_id"]), ancestry)
        except InvalidContinuationCheckpoint:
            raise
        except (TypeError, ValueError, KeyError, binascii.Error, json.JSONDecodeError, InvalidOrganismCheckpoint) as exc:
            raise InvalidContinuationCheckpoint("continuation checkpoint is malformed") from exc


@dataclass(slots=True)
class _CanonicalRecord:
    continuation_id: ContinuationId
    organism_step: int
    state_sha256: str


class _RegistryAuthority:
    """Live process-local state for one serialized owner-store lineage."""

    def __init__(self, records: dict[OrganismIdentity, _CanonicalRecord] | None = None) -> None:
        self.records = records if records is not None else {}
        self.active: set[OrganismIdentity] = set()
        self.lock = RLock()


_authority_cache_lock = RLock()
_authority_cache: WeakValueDictionary[UUID, _RegistryAuthority] = WeakValueDictionary()
# Retain only a compact fingerprint after a live registry is collected. This
# prevents an old snapshot from recreating stale process-local authority without
# retaining every registry's records for the process lifetime.
_authority_fingerprints: dict[UUID, str] = {}


def _records_fingerprint(records: dict[OrganismIdentity, _CanonicalRecord]) -> str:
    canonical = [
        [str(identity), str(record.continuation_id), record.organism_step, record.state_sha256]
        for identity, record in sorted(records.items(), key=lambda item: str(item[0]))
    ]
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return _sha256(payload)


def _attach_registry_authority(
    authority_id: UUID,
    records: dict[OrganismIdentity, _CanonicalRecord] | None = None,
) -> _RegistryAuthority:
    """Share live state for repeated restores of the same store in this process."""
    with _authority_cache_lock:
        existing = _authority_cache.get(authority_id)
        if existing is None:
            restored_records = records if records is not None else {}
            fingerprint = _records_fingerprint(restored_records)
            known_fingerprint = _authority_fingerprints.get(authority_id)
            if known_fingerprint is not None and fingerprint != known_fingerprint:
                raise StaleCanonicalContinuation(
                    "serialized registry does not match current process authority state"
                )
            _authority_fingerprints[authority_id] = fingerprint
            authority = _RegistryAuthority(restored_records)
            _authority_cache[authority_id] = authority
            return authority

    # Do not acquire an authority lock while holding the cache lock: state
    # mutations take the authority lock before refreshing the fingerprint.
    if records is not None:
        with existing.lock:
            if existing.records != records:
                raise StaleCanonicalContinuation(
                    "serialized registry does not match current process authority state"
                )
    return existing


def _remember_registry_authority(authority_id: UUID, authority: _RegistryAuthority) -> None:
    with _authority_cache_lock:
        _authority_fingerprints[authority_id] = _records_fingerprint(authority.records)


class CanonicalContinuationRegistry:
    """Process-local canonical registry with one live lease per store lineage.

    Repeated restores of one serialized owner store share live state in this
    Python process. A compact fingerprint remains after collection so stale
    snapshots cannot recreate old authority. The durable owner store must still
    perform atomic compare-and-commit for cross-process safety.
    """

    def __init__(self) -> None:
        self._authority_id = uuid4()
        self._authority = _attach_registry_authority(self._authority_id)

    SCHEMA = "biopeia.organism-canonical-continuations"
    SCHEMA_VERSION = 2

    def to_bytes(self) -> bytes:
        """Serialize owner registry state, not a portable authority token."""
        with self._authority.lock:
            records = [
                {
                    "organism_id": str(identity),
                    "continuation_id": str(record.continuation_id),
                    "organism_step": record.organism_step,
                    "state_sha256": record.state_sha256,
                }
                for identity, record in sorted(self._authority.records.items(), key=lambda item: str(item[0]))
            ]
            _remember_registry_authority(self._authority_id, self._authority)
        return json.dumps(
            {
                "schema": self.SCHEMA,
                "schema_version": self.SCHEMA_VERSION,
                "authority_id": str(self._authority_id),
                "records": records,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "CanonicalContinuationRegistry":
        """Restore owner registry state; caller must load it from its authority store."""
        try:
            doc = json.loads(payload)
            if not isinstance(doc, dict) or set(doc) != {
                "schema", "schema_version", "authority_id", "records"
            }:
                raise InvalidContinuationCheckpoint("canonical registry has an invalid shape")
            if doc["schema"] != cls.SCHEMA or type(doc["schema_version"]) is not int or doc["schema_version"] != cls.SCHEMA_VERSION:
                raise InvalidContinuationCheckpoint("unsupported canonical registry schema")
            if not isinstance(doc["records"], list):
                raise InvalidContinuationCheckpoint("canonical registry records must be a list")
            authority_id = UUID(doc["authority_id"])
            records: dict[OrganismIdentity, _CanonicalRecord] = {}
            for raw in doc["records"]:
                if not isinstance(raw, dict) or set(raw) != {
                    "organism_id", "continuation_id", "organism_step", "state_sha256"
                }:
                    raise InvalidContinuationCheckpoint("canonical registry record has an invalid shape")
                identity = OrganismIdentity.parse(raw["organism_id"])
                step = raw["organism_step"]
                digest = raw["state_sha256"]
                if type(step) is not int or step < 0:
                    raise InvalidContinuationCheckpoint("canonical registry step must be non-negative")
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise InvalidContinuationCheckpoint("canonical registry digest must be SHA-256")
                if identity in records:
                    raise InvalidContinuationCheckpoint("duplicate Organism identity in canonical registry")
                records[identity] = _CanonicalRecord(
                    ContinuationId.parse(raw["continuation_id"]), step, digest
                )
            registry = cls.__new__(cls)
            registry._authority_id = authority_id
            registry._authority = _attach_registry_authority(authority_id, records)
            return registry
        except (InvalidContinuationCheckpoint, StaleCanonicalContinuation):
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise InvalidContinuationCheckpoint("canonical registry is malformed") from exc

    def register_genesis(self, candidate: OrganismContinuation) -> None:
        """Install an explicitly authorized initial continuation."""
        with self._authority.lock:
            if candidate.identity in self._authority.records:
                raise DuplicateCanonicalContinuation("Organism identity is already registered")
            if candidate.ancestry is not None:
                raise ContinuationError("genesis continuation cannot carry branch ancestry")
            if candidate.organism_step != 0:
                raise ContinuationError("genesis continuation must start at Organism step zero")
            self._authority.records[candidate.identity] = self._record(candidate)
            _remember_registry_authority(self._authority_id, self._authority)

    def designate_branch_from(
        self, candidate: OrganismContinuation, *, source: OrganismContinuation
    ) -> None:
        """Admit a branch whose declared ancestry matches a supplied source.

        This checks consistency, not authenticity. Durable owner storage must
        authorize the source and atomically record the resulting identity.
        """
        with self._authority.lock:
            ancestry = candidate.ancestry
            if ancestry is None:
                raise ContinuationError("branch designation requires explicit ancestry")
            if ancestry.parent_organism_id != source.identity:
                raise ContinuationError("branch ancestry parent identity does not match source")
            if ancestry.parent_continuation_id != source.continuation_id:
                raise ContinuationError("branch ancestry continuation does not match source")
            if ancestry.parent_organism_step != source.organism_step:
                raise ContinuationError("branch ancestry step does not match source")
            if ancestry.source_checkpoint_sha256 != _sha256(source.state.owner_checkpoint.to_bytes()):
                raise ContinuationError("branch ancestry checkpoint digest does not match source")
            if candidate.identity in self._authority.records:
                raise DuplicateCanonicalContinuation("branch Organism identity is already registered")
            self._authority.records[candidate.identity] = self._record(candidate)
            _remember_registry_authority(self._authority_id, self._authority)

    def acquire(self, candidate: OrganismContinuation) -> "CanonicalContinuationLease":
        """Open only the exact latest state and refuse duplicate live restores."""
        with self._authority.lock:
            record = self._authority.records.get(candidate.identity)
            if record is None or not self._matches(record, candidate):
                raise StaleCanonicalContinuation("checkpoint is not the current canonical continuation state")
            if candidate.identity in self._authority.active:
                raise DuplicateCanonicalContinuation("canonical continuation is already live")
            self._authority.active.add(candidate.identity)
            return CanonicalContinuationLease(self, candidate)

    def _commit(self, current: OrganismContinuation, proposed: OrganismContinuation) -> None:
        with self._authority.lock:
            record = self._authority.records.get(current.identity)
            if current.identity not in self._authority.active or record is None or not self._matches(record, current):
                raise StaleCanonicalContinuation("expected canonical continuation is no longer current")
            if proposed.identity != current.identity or proposed.continuation_id != current.continuation_id:
                raise ContinuationError("canonical advancement must preserve identity and continuation_id")
            if proposed.ancestry != current.ancestry:
                raise ContinuationError("canonical advancement cannot alter continuation ancestry")
            ordinary = (
                current.state.lifecycle is OrganismLifecycle.ACTIVE
                and proposed.state.lifecycle is OrganismLifecycle.ACTIVE
                and proposed.organism_step == current.organism_step + 1
                and proposed.state.lifecycle == current.state.lifecycle
                and proposed.state.lifecycle_history == current.state.lifecycle_history
                and proposed.state.embodiment == current.state.embodiment
            )
            reembodiment_reactivation = (
                current.state.lifecycle is OrganismLifecycle.DORMANT
                and proposed.state.lifecycle is OrganismLifecycle.ACTIVE
                and proposed.state.lifecycle_history[:-1] == current.state.lifecycle_history
                and proposed.state.lifecycle_history[-1].from_state is OrganismLifecycle.DORMANT
                and proposed.state.lifecycle_history[-1].to_state is OrganismLifecycle.ACTIVE
            )
            lifecycle_only = (
                proposed.organism_step == current.organism_step
                and proposed.state.lifecycle != current.state.lifecycle
                and proposed.state.lifecycle_history[:-1] == current.state.lifecycle_history
                and proposed.state.owner_checkpoint == current.state.owner_checkpoint
                and (proposed.state.embodiment == current.state.embodiment or reembodiment_reactivation)
                and proposed.state.world_correlation == current.state.world_correlation
            )
            if not (ordinary or lifecycle_only):
                raise ContinuationError("canonical commit must be one active step or one lifecycle transition")
            self._authority.records[current.identity] = self._record(proposed)
            _remember_registry_authority(self._authority_id, self._authority)

    def _release(self, identity: OrganismIdentity) -> None:
        with self._authority.lock:
            self._authority.active.discard(identity)

    @staticmethod
    def _record(candidate: OrganismContinuation) -> _CanonicalRecord:
        from hashlib import sha256

        return _CanonicalRecord(candidate.continuation_id, candidate.organism_step, sha256(candidate.to_bytes()).hexdigest())

    @staticmethod
    def _matches(record: _CanonicalRecord, candidate: OrganismContinuation) -> bool:
        return (
            record.continuation_id == candidate.continuation_id
            and record.organism_step == candidate.organism_step
            and record.state_sha256 == CanonicalContinuationRegistry._record(candidate).state_sha256
        )


class CanonicalContinuationLease:
    """Exclusive process-local handle to one canonical continuation."""

    def __init__(self, registry: CanonicalContinuationRegistry, current: OrganismContinuation) -> None:
        self._registry = registry
        self.current = current
        self._closed = False

    def commit(self, proposed: OrganismContinuation) -> None:
        if self._closed:
            raise ContinuationError("continuation lease is closed")
        self._registry._commit(self.current, proposed)
        self.current = proposed

    def close(self) -> None:
        if not self._closed:
            self._registry._release(self.current.identity)
            self._closed = True

    def __enter__(self) -> "CanonicalContinuationLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def branch_continuation(
    source: OrganismContinuation,
    *,
    new_identity: OrganismIdentity,
    new_continuation_id: ContinuationId,
    branched_state: OrganismTransitionState,
) -> OrganismContinuation:
    """Create a provenance-linked fork using owner-produced private state.

    This function never rewrites opaque private bytes or asserts that the fork
    is canonical. The Organism owner must produce state for the new identity;
    a separate explicit registry designation is required for canonical use.
    """
    if (
        new_identity == source.identity
        or branched_state.identity != new_identity
        or new_continuation_id == source.continuation_id
    ):
        raise ContinuationError("a branch must use owner-produced state under a new Organism identity")
    return OrganismContinuation(
        state=branched_state,
        continuation_id=new_continuation_id,
        ancestry=ContinuationAncestry(
            parent_organism_id=source.identity,
            parent_continuation_id=source.continuation_id,
            parent_organism_step=source.organism_step,
            source_checkpoint_sha256=_sha256(source.state.owner_checkpoint.to_bytes()),
        ),
    )


def _sha256(payload: bytes) -> str:
    from hashlib import sha256

    return sha256(payload).hexdigest()
