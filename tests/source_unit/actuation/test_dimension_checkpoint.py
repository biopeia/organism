from __future__ import annotations

from copy import deepcopy
import json

import pytest

from organism.actuation.dimension import ActionDimensionRegistry, opaque_dimension_id


def _checkpoint() -> dict[str, object]:
    channels = ["opaque.channel.a", "opaque.channel.b"]
    return {
        "schema_version": 2,
        "capacity": 4,
        "items": [
            {
                "dimension_id": opaque_dimension_id(channels),
                "intervention_signature_refs": ["intervention.signature.1"],
                "channel_refs": channels,
                "verified_surfaces": ["surface.1"],
                "controllability": 0.75,
                "confidence": 0.5,
                "usage_count": 3,
            }
        ],
    }


def test_action_dimension_registry_checkpoint_round_trips_and_rebinds() -> None:
    payload = json.loads(json.dumps(_checkpoint()))
    restored = ActionDimensionRegistry.restore(payload)

    assert restored.checkpoint() == payload
    assert restored.items[0].availability is False
    assert restored.items[0].embodiment_bound is False

    restored.bind_surface(
        ("opaque.channel.a", "opaque.channel.b"), surface_fingerprint="surface.1"
    )
    assert restored.items[0].availability is True
    assert restored.items[0].embodiment_bound is True


@pytest.mark.parametrize("payload", [None, [], "checkpoint", {}])
def test_restore_rejects_absent_or_non_mapping_checkpoint(payload: object) -> None:
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore(payload)  # type: ignore[arg-type]


def test_restore_requires_exact_top_level_shape() -> None:
    valid = _checkpoint()
    for key in tuple(valid):
        missing = deepcopy(valid)
        del missing[key]
        with pytest.raises(ValueError):
            ActionDimensionRegistry.restore(missing)
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore({**valid, "future": None})


@pytest.mark.parametrize("version", [True, "2", None])
def test_restore_rejects_invalid_schema_version(version: object) -> None:
    payload = _checkpoint()
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore(payload)


@pytest.mark.parametrize("capacity", [True, 1.5, "4", 0, -1])
def test_restore_rejects_invalid_capacity(capacity: object) -> None:
    payload = _checkpoint()
    payload["capacity"] = capacity
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dimension_id", 1),
        ("intervention_signature_refs", None),
        ("intervention_signature_refs", ["not-a-signature"]),
        ("channel_refs", None),
        ("channel_refs", ["opaque.channel.a", "opaque.channel.a"]),
        ("verified_surfaces", []),
        ("verified_surfaces", ["surface.1", "surface.1"]),
        ("controllability", True),
        ("controllability", float("nan")),
        ("controllability", 1.1),
        ("confidence", "0.5"),
        ("confidence", float("inf")),
        ("usage_count", True),
        ("usage_count", -1),
    ],
)
def test_restore_rejects_malformed_entry_values(field: str, value: object) -> None:
    payload = _checkpoint()
    entry = payload["items"][0]  # type: ignore[index]
    entry[field] = value
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore(payload)


def test_restore_rejects_unknown_or_missing_entry_fields() -> None:
    payload = _checkpoint()
    entry = payload["items"][0]  # type: ignore[index]
    del entry["confidence"]
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore(payload)

    payload = _checkpoint()
    entry = payload["items"][0]  # type: ignore[index]
    entry["unknown"] = None
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore(payload)


def test_restore_rejects_duplicate_dimensions_and_over_capacity_payloads() -> None:
    payload = _checkpoint()
    entry = payload["items"][0]  # type: ignore[index]
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore({**payload, "items": [entry, deepcopy(entry)]})

    second = {
        **deepcopy(entry),
        "dimension_id": opaque_dimension_id(["opaque.channel.c"]),
        "channel_refs": ["opaque.channel.c"],
    }
    with pytest.raises(ValueError):
        ActionDimensionRegistry.restore(
            {**payload, "capacity": 1, "items": [entry, second]}
        )

