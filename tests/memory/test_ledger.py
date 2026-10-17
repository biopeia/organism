from __future__ import annotations

import copy
import json

import pytest

from organism.memory import (
    EpistemicStatus,
    ExperienceLedger,
    ExperienceRecord,
    SourceKind,
)


def _record(record_id: str, organism_id: str = "org-1") -> ExperienceRecord:
    return ExperienceRecord(
        record_id=record_id,
        organism_id=organism_id,
        tick_class=int(record_id.removeprefix("r")),
        context_tokens=("context",),
        action_token="act",
        outcome_tokens=("outcome",),
        epistemic_status=EpistemicStatus.HYPOTHESIZED,
        evidence_refs=(),
        confidence_class=2,
        source_kind=SourceKind.INTERNAL,
    )


def test_ledger_round_trip_preserves_order_and_organism_identity() -> None:
    ledger = ExperienceLedger("org-1", max_records=16)
    ledger.extend((_record("r1"), _record("r2")))

    restored = ExperienceLedger.restore(ledger.checkpoint(), organism_id="org-1")

    assert restored.organism_id == "org-1"
    assert restored.records == ledger.records
    assert restored.get("r1") == ledger.records[0]


def test_ledger_is_bounded_and_reports_evicted_oldest_record() -> None:
    ledger = ExperienceLedger("org-1", max_records=16)
    records = [_record(f"r{i}") for i in range(16)]
    ledger.extend(records)

    evicted = ledger.append(_record("r16"))

    assert evicted == records[0]
    assert len(ledger.records) == 16
    assert ledger.get("r0") is None
    assert ledger.get("r16") == _record("r16")


def test_ledger_rejects_foreign_and_duplicate_record_identity() -> None:
    ledger = ExperienceLedger("org-1", max_records=16)
    ledger.append(_record("r1"))

    with pytest.raises(ValueError, match="different organism"):
        ledger.append(_record("r2", "org-2"))
    with pytest.raises(ValueError, match="already exists"):
        ledger.append(_record("r1"))


def test_restore_rejects_foreign_identity_duplicate_records_and_overflow() -> None:
    payload = ExperienceLedger("org-1", max_records=16)
    payload.append(_record("r1"))
    valid = payload.checkpoint()

    with pytest.raises(ValueError, match="organism mismatch"):
        ExperienceLedger.restore(valid, organism_id="org-2")

    duplicate = copy.deepcopy(valid)
    duplicate["records"].append(duplicate["records"][0])  # type: ignore[index]
    with pytest.raises(ValueError, match="already exists"):
        ExperienceLedger.restore(duplicate, organism_id="org-1")

    overflow = copy.deepcopy(valid)
    overflow["records"] = [  # type: ignore[index]
        _record(f"r{i}").canonical_payload() for i in range(17)
    ]
    with pytest.raises(ValueError, match="invalid experience ledger records"):
        ExperienceLedger.restore(overflow, organism_id="org-1")


def test_restore_rejects_missing_or_null_required_checkpoint_state() -> None:
    valid = ExperienceLedger("org-1", max_records=16).checkpoint()

    with pytest.raises(ValueError, match="invalid experience ledger checkpoint"):
        ExperienceLedger.restore(None, organism_id="org-1")  # type: ignore[arg-type]

    missing_records = dict(valid)
    del missing_records["records"]
    with pytest.raises(ValueError, match="invalid experience ledger checkpoint"):
        ExperienceLedger.restore(missing_records, organism_id="org-1")

    null_records = {**valid, "records": None}
    with pytest.raises(ValueError, match="invalid experience ledger records"):
        ExperienceLedger.restore(null_records, organism_id="org-1")

    record = _record("r1").canonical_payload()
    del record["context_tokens"]
    invalid_record = {**valid, "records": [record]}
    with pytest.raises(ValueError, match="experience checkpoint fields"):
        ExperienceLedger.restore(invalid_record, organism_id="org-1")


def test_byte_checkpoint_round_trip_preserves_non_default_capacity_and_order() -> None:
    ledger = ExperienceLedger("org-1", max_records=37)
    ledger.extend((_record("r1"), _record("r2")))

    encoded = ledger.to_bytes()
    restored = ExperienceLedger.from_bytes(encoded, organism_id="org-1")

    assert isinstance(encoded, bytes)
    assert restored.checkpoint() == ledger.checkpoint()
    assert restored.to_bytes() == encoded


def test_byte_checkpoint_restoration_preserves_next_ledger_transition() -> None:
    continuous = ExperienceLedger("org-1", max_records=16)
    continuous.extend((_record("r1"), _record("r2")))
    restored = ExperienceLedger.from_bytes(continuous.to_bytes(), organism_id="org-1")

    continuous.append(_record("r3"))
    restored.append(_record("r3"))

    assert restored.checkpoint() == continuous.checkpoint()
    assert restored.to_bytes() == continuous.to_bytes()


@pytest.mark.parametrize(
    "encoded",
    [
        b"{",
        b"[]",
        b'{"schema_version":1,"schema_version":1}',
        json.dumps(
            {
                "schema_version": 1,
                "organism_id": "org-1",
                "max_records": 37,
                "records": [],
                "unexpected": True,
            }
        ).encode(),
    ],
)
def test_byte_checkpoint_rejects_malformed_or_ambiguous_payloads(encoded: bytes) -> None:
    with pytest.raises(ValueError, match="invalid experience ledger checkpoint"):
        ExperienceLedger.from_bytes(encoded, organism_id="org-1")


@pytest.mark.parametrize("encoded", [None, "{}", bytearray(b"{}")])
def test_byte_checkpoint_rejects_non_bytes_payloads(encoded: object) -> None:
    with pytest.raises(ValueError, match="invalid experience ledger checkpoint bytes"):
        ExperienceLedger.from_bytes(encoded, organism_id="org-1")  # type: ignore[arg-type]


def test_byte_checkpoint_requires_matching_owner_identity() -> None:
    encoded = ExperienceLedger("org-1", max_records=37).to_bytes()

    with pytest.raises(ValueError, match="organism mismatch"):
        ExperienceLedger.from_bytes(encoded, organism_id="org-2")
