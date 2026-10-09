from __future__ import annotations

import copy

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
