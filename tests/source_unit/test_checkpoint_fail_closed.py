from __future__ import annotations

import copy

import pytest

from organism.capacity import CapacityPressure
from organism.provenance import CausalEvent, CausalRef, ProvenanceLog


def test_capacity_pressure_checkpoint_requires_complete_matching_state() -> None:
    pressure = CapacityPressure(8)
    pressure.note_evicted(("item.a",))
    checkpoint = pressure.checkpoint()

    assert CapacityPressure.restore(checkpoint, capacity=8).checkpoint() == checkpoint
    for invalid in (
        None,
        {},
        {**checkpoint, "admissions": None},
        {**checkpoint, "schema_version": True},
    ):
        with pytest.raises(ValueError):
            CapacityPressure.restore(invalid, capacity=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capacity mismatch"):
        CapacityPressure.restore(checkpoint, capacity=9)


def test_provenance_checkpoint_requires_complete_frontier() -> None:
    log = ProvenanceLog(capacity=8)
    log.emit(
        # The frontier is the owner-held continuation state; the event ring is
        # intentionally not part of this bounded checkpoint format.
        CausalEvent(
            tick=1,
            domain="organism",
            operation="test",
            subject=CausalRef("state", "one"),
            caused_by=(),
        )
    )
    checkpoint = log.checkpoint()

    assert ProvenanceLog.restore(checkpoint).checkpoint() == checkpoint
    for invalid in (
        None,
        {},
        {**checkpoint, "frontier": None},
        {**checkpoint, "schema_version": True},
    ):
        with pytest.raises(ValueError):
            ProvenanceLog.restore(invalid)  # type: ignore[arg-type]


def test_provenance_frontier_may_contain_multiple_outputs_from_one_event() -> None:
    log = ProvenanceLog()
    log.emit(
        CausalEvent(
            tick=1,
            domain="organism",
            operation="test",
            subject=CausalRef("state", "one"),
            caused_by=(),
            produced=(CausalRef("state", "one"), CausalRef("state", "two")),
        )
    )
    checkpoint = log.checkpoint()

    assert len(checkpoint["frontier"]) == 2
    assert ProvenanceLog.restore(checkpoint).checkpoint() == checkpoint


def test_provenance_checkpoint_rejects_malformed_and_duplicate_frontier_refs() -> None:
    log = ProvenanceLog()
    log.emit(
        CausalEvent(
            tick=1,
            domain="organism",
            operation="test",
            subject=CausalRef("state", "one"),
            caused_by=(),
        )
    )
    valid = log.checkpoint()

    malformed = copy.deepcopy(valid)
    malformed["frontier"][0]["caused_by"] = None  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceLog.restore(malformed)

    duplicate = copy.deepcopy(valid)
    duplicate["frontier"].append(duplicate["frontier"][0])  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate"):
        ProvenanceLog.restore(duplicate)


def test_provenance_checkpoint_rejects_malformed_frontier_event_id() -> None:
    log = ProvenanceLog()
    log.emit(
        CausalEvent(
            tick=1,
            domain="organism",
            operation="test",
            subject=CausalRef("state", "one"),
            caused_by=(),
        )
    )
    malformed = copy.deepcopy(log.checkpoint())
    malformed["frontier"][0]["event_id"] = "not-an-event-id"  # type: ignore[index]

    with pytest.raises(ValueError, match="frontier entry"):
        ProvenanceLog.restore(malformed)


def test_causal_event_payload_round_trips_complete_serialized_state() -> None:
    event = CausalEvent(
        tick=4,
        domain="organism",
        operation="transition",
        subject=CausalRef("state", "one"),
        caused_by=(CausalRef("input", "two"),),
        produced=(CausalRef("state", "three"),),
        rule="rule.v1",
        parameters={"gain": 0.25, "enabled": True, "label": "bounded"},
    )

    assert CausalEvent.from_payload(event.payload()).payload() == event.payload()


def test_causal_event_payload_round_trips_explicit_null_rule() -> None:
    event = CausalEvent(
        tick=4,
        domain="organism",
        operation="transition",
        subject=CausalRef("state", "one"),
        caused_by=(),
    )

    payload = event.payload()
    assert payload["rule"] is None
    assert CausalEvent.from_payload(payload).payload() == payload


def test_causal_event_payload_rejects_missing_or_unknown_serialized_fields() -> None:
    event = CausalEvent(
        tick=4,
        domain="organism",
        operation="transition",
        subject=CausalRef("state", "one"),
        caused_by=(),
    )
    payload = event.payload()

    for field in payload:
        missing = dict(payload)
        del missing[field]
        with pytest.raises(ValueError, match="payload fields"):
            CausalEvent.from_payload(missing)

    with pytest.raises(ValueError, match="payload fields"):
        CausalEvent.from_payload({**payload, "unknown": "field"})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tick", "4"),
        ("tick", True),
        ("domain", 1),
        ("operation", None),
        ("subject", ("state", "one")),
        ("caused_by", ()),
        ("caused_by", [("input", "two")]),
        ("produced", ()),
        ("rule", 7),
        ("parameters", []),
        ("parameters", {"invalid": []}),
        ("event_id", None),
        ("event_id", "event.stale"),
    ),
)
def test_causal_event_payload_rejects_malformed_serialized_field(
    field: str, value: object
) -> None:
    event = CausalEvent(
        tick=4,
        domain="organism",
        operation="transition",
        subject=CausalRef("state", "one"),
        caused_by=(),
    )
    payload = event.payload()
    payload[field] = value

    with pytest.raises(ValueError):
        CausalEvent.from_payload(payload)
