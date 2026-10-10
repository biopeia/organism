from __future__ import annotations

from copy import deepcopy
import json

import pytest

from organism.actuation.action import MotorCommand
from organism.actuation.intervention import InterventionSignatureRegistry


def _recorded_registry() -> InterventionSignatureRegistry:
    registry = InterventionSignatureRegistry(capacity=4)
    command = MotorCommand.from_mapping(
        command_id="command-1",
        commitment_id="commitment-1",
        controller_id="controller-1",
        competence_id="competence-1",
        surface_fingerprint="surface-1",
        channels={"actuator-a": 0.5},
        issued_at_tick=3,
    )
    registry.signature_for_command(command=command, controller_id="controller-1", tick=3)
    registry.signature_for_command(
        command=MotorCommand.from_mapping(
            command_id="command-2",
            commitment_id="commitment-1",
            controller_id="controller-1",
            competence_id="competence-1",
            surface_fingerprint="surface-1",
            channels={"actuator-a": 0.75},
            issued_at_tick=4,
        ),
        controller_id="controller-1",
        tick=4,
    )
    return registry


def test_intervention_registry_checkpoint_restores_and_continues_command_evidence() -> None:
    registry = _recorded_registry()
    persisted = json.loads(json.dumps(registry.checkpoint()))

    restored = InterventionSignatureRegistry.restore(persisted)

    assert restored.checkpoint() == registry.checkpoint()
    assert restored.items == registry.items
    assert restored.recurring_count == registry.recurring_count
    assert restored.attempt_support(registry.items[0].signature_id) == 2
    assert restored.intensity_coverage(registry.items[0].signature_id) == 0.5

    continued = MotorCommand.from_mapping(
        command_id="command-after-restore",
        commitment_id="commitment-1",
        controller_id="controller-1",
        competence_id="competence-1",
        surface_fingerprint="surface-1",
        channels={"actuator-a": 0.25},
        issued_at_tick=5,
    )
    signature = restored.signature_for_command(
        command=continued,
        controller_id="controller-1",
        tick=5,
    )

    assert restored.attempt_support(signature.signature_id) == 3
    assert restored.intensity_coverage(signature.signature_id) == 0.75


def test_intervention_registry_preserves_temporal_signature_statistics() -> None:
    registry = _recorded_registry()
    temporal = registry.signature_for_sequence(
        command_refs=("command-1", "command-2"),
        controller_seed_ref=None,
        tick=5,
    )

    restored = InterventionSignatureRegistry.restore(
        json.loads(json.dumps(registry.checkpoint()))
    )

    assert restored.get(temporal.signature_id) == temporal
    assert restored.attempt_support(temporal.signature_id) == 1
    assert restored.intensity_coverage(temporal.signature_id) == 0.0


def test_intervention_registry_restore_requires_exact_top_level_schema() -> None:
    valid = _recorded_registry().checkpoint()
    for key in tuple(valid):
        missing = deepcopy(valid)
        del missing[key]
        with pytest.raises(ValueError):
            InterventionSignatureRegistry.restore(missing)

    with pytest.raises(ValueError):
        InterventionSignatureRegistry.restore({**valid, "future": None})
    for invalid in (None, [], "checkpoint"):
        with pytest.raises(ValueError):
            InterventionSignatureRegistry.restore(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("version", [True, 2, "1", None])
def test_intervention_registry_restore_rejects_unsupported_schema(version: object) -> None:
    payload = _recorded_registry().checkpoint()
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        InterventionSignatureRegistry.restore(payload)


@pytest.mark.parametrize("capacity", [True, 1.5, "4", 0, -1])
def test_intervention_registry_restore_rejects_invalid_capacity(capacity: object) -> None:
    payload = _recorded_registry().checkpoint()
    payload["capacity"] = capacity
    with pytest.raises(ValueError):
        InterventionSignatureRegistry.restore(payload)


def test_intervention_registry_restore_rejects_invalid_signatures_collection() -> None:
    registry = _recorded_registry()
    registry.signature_for_command(
        command=MotorCommand.from_mapping(
            command_id="command-3",
            commitment_id="commitment-1",
            controller_id="controller-1",
            competence_id="competence-1",
            surface_fingerprint="surface-1",
            channels={"actuator-b": 0.5},
            issued_at_tick=5,
        ),
        controller_id="controller-1",
        tick=5,
    )
    valid = registry.checkpoint()
    entry = valid["signatures"][0]
    for signatures in (None, {}, [None], [entry, entry]):
        with pytest.raises(ValueError):
            InterventionSignatureRegistry.restore({**valid, "signatures": signatures})

    over_capacity = {**valid, "capacity": 1}
    with pytest.raises(ValueError):
        InterventionSignatureRegistry.restore(over_capacity)

    with pytest.raises(ValueError):
        InterventionSignatureRegistry.restore(
            {**valid, "signatures": [{**entry, "unknown": True}]}
        )


def test_intervention_registry_restore_rejects_invalid_entry_fields_and_values() -> None:
    valid = _recorded_registry().checkpoint()
    entry = valid["signatures"][0]
    invalid_entries = (
        {key: value for key, value in entry.items() if key != "attempts"},
        {**entry, "signature_id": 1},
        {**entry, "signature_id": "wrong-prefix"},
        {**entry, "channel_refs": None},
        {**entry, "channel_refs": ["not-opaque"]},
        {**entry, "temporal_pattern_ref": "not-opaque"},
        {**entry, "attempts": True},
        {**entry, "attempts": 0},
        {**entry, "first_tick": -1},
        {**entry, "last_tick": 2.5},
        {**entry, "last_tick": entry["first_tick"] - 1},
        {**entry, "intensity_counts": [0, 1]},
        {**entry, "intensity_counts": [0, 0, True, 0]},
        {**entry, "intensity_counts": [0, 0, -1, 0]},
        {**entry, "intensity_counts": [0, 0, 0, 0, 0]},
    )
    for invalid in invalid_entries:
        with pytest.raises(ValueError):
            InterventionSignatureRegistry.restore({**valid, "signatures": [invalid]})


def test_intervention_registry_restore_rejects_inconsistent_statistics() -> None:
    valid = _recorded_registry().checkpoint()
    entry = valid["signatures"][0]

    invalid_counts = {**entry, "intensity_counts": [0, 0, 0, 0]}
    with pytest.raises(ValueError):
        InterventionSignatureRegistry.restore({**valid, "signatures": [invalid_counts]})

    invalid_temporal = {
        **entry,
        "signature_id": "intervention.signature.temporal",
        "temporal_pattern_ref": "temporal.sequence",
        "intensity_counts": [0, 1, 0, 0],
    }
    with pytest.raises(ValueError):
        InterventionSignatureRegistry.restore({**valid, "signatures": [invalid_temporal]})
