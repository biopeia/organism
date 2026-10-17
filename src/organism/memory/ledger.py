from __future__ import annotations

from collections import deque
import json
from typing import Iterable, Mapping

from .experience import ExperienceRecord


class ExperienceLedger:
    """Bounded immutable episode ledger owned by one organism.

    Records are append-only by identity. When capacity is reached the oldest
    episode ages out; the ledger never rewrites an episode in place. Evidence
    correction is represented by a new record with its own epistemic state and
    provenance, preserving the history needed for scientific audit.
    """

    SCHEMA_VERSION = 1

    def __init__(self, organism_id: str, *, max_records: int = 2048) -> None:
        if not isinstance(organism_id, str) or not organism_id or len(organism_id) > 128:
            raise ValueError("organism_id must be a bounded non-empty string")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or not 16 <= max_records <= 8192
        ):
            raise ValueError("max_records must be within [16, 8192]")
        self._organism_id = organism_id
        self._max_records = max_records
        self._records: deque[ExperienceRecord] = deque(maxlen=max_records)
        self._record_ids: set[str] = set()

    @property
    def organism_id(self) -> str:
        return self._organism_id

    @property
    def max_records(self) -> int:
        return self._max_records

    @property
    def records(self) -> tuple[ExperienceRecord, ...]:
        return tuple(self._records)

    def get(self, record_id: str) -> ExperienceRecord | None:
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def append(self, record: ExperienceRecord) -> ExperienceRecord | None:
        if not isinstance(record, ExperienceRecord):
            raise ValueError("record must be an ExperienceRecord")
        if record.organism_id != self._organism_id:
            raise ValueError("experience record belongs to a different organism")
        if record.record_id in self._record_ids:
            raise ValueError("experience record_id already exists")
        evicted = self._records[0] if len(self._records) == self._max_records else None
        self._records.append(record)
        self._record_ids.add(record.record_id)
        if evicted is not None:
            self._record_ids.discard(evicted.record_id)
        return evicted

    def extend(self, records: Iterable[ExperienceRecord]) -> None:
        for record in records:
            self.append(record)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "organism_id": self._organism_id,
            "max_records": self._max_records,
            "records": [record.canonical_payload() for record in self._records],
        }

    def to_bytes(self) -> bytes:
        """Serialize this owner state deterministically for a checkpoint envelope."""
        return json.dumps(
            self.checkpoint(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes, *, organism_id: str) -> "ExperienceLedger":
        """Restore an exact ledger checkpoint, rejecting ambiguous JSON objects."""
        if not isinstance(payload, bytes):
            raise ValueError("invalid experience ledger checkpoint bytes")

        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result

        try:
            decoded = json.loads(
                payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid experience ledger checkpoint bytes") from exc
        if not isinstance(decoded, dict):
            raise ValueError("invalid experience ledger checkpoint bytes")
        return cls.restore(decoded, organism_id=organism_id)

    @classmethod
    def restore(
        cls, payload: Mapping[str, object], *, organism_id: str
    ) -> "ExperienceLedger":
        required = {"schema_version", "organism_id", "max_records", "records"}
        if (
            not isinstance(payload, Mapping)
            or set(payload) != required
            or isinstance(payload.get("schema_version"), bool)
            or not isinstance(payload.get("schema_version"), int)
            or payload.get("schema_version") != cls.SCHEMA_VERSION
        ):
            raise ValueError("invalid experience ledger checkpoint")
        if payload.get("organism_id") != organism_id:
            raise ValueError("experience ledger organism mismatch")
        raw_max = payload["max_records"]
        if isinstance(raw_max, bool) or not isinstance(raw_max, int):
            raise ValueError("invalid experience ledger capacity")
        ledger = cls(organism_id, max_records=raw_max)
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or len(raw_records) > ledger.max_records:
            raise ValueError("invalid experience ledger records")
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise ValueError("experience ledger record must be an object")
            ledger.append(ExperienceRecord.restore(raw))
        return ledger
