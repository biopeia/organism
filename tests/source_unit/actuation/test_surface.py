from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from organism.actuation.surface import (
    ActuatorSurface,
    MotorSlot,
    derive_actuator_constitution,
)


def test_derive_actuator_constitution_produces_body_owned_slots():
    effectors = ("fixture.effector.left", "fixture.effector.right", "fixture.effector.top", "fixture.effector.bottom")
    constitution = derive_actuator_constitution(effectors)
    assert len(constitution.slots) == 4
    assert all(isinstance(slot, MotorSlot) for slot in constitution.slots)


def test_effector_identity_is_stable_when_input_order_changes():
    first = derive_actuator_constitution(
        ("body.left", "body.right"), fingerprint_material="body-a"
    )
    permuted = derive_actuator_constitution(
        ("body.right", "body.left"), fingerprint_material="body-a"
    )
    assert first.slot_for("body.left").actuator_id == permuted.slot_for("body.left").actuator_id
    assert first.slot_for("body.right").actuator_id == permuted.slot_for("body.right").actuator_id
    assert first.contract_fingerprint == permuted.contract_fingerprint
    assert first.actuator_ids == ("body.left", "body.right")
    assert permuted.actuator_ids == ("body.right", "body.left")


def test_effector_ids_are_unique_and_preserved_verbatim():
    first = derive_actuator_constitution(
        ("body.left", "body.right"), fingerprint_material="body-a"
    )
    second = derive_actuator_constitution(
        ("body.left", "body.right"), fingerprint_material="body-a"
    )
    assert first.actuator_ids == second.actuator_ids
    assert first.contract_fingerprint == second.contract_fingerprint
    assert first.actuator_ids == ("body.left", "body.right")


def test_contract_metadata_changes_fingerprint_not_effector_identity():
    first = derive_actuator_constitution(
        ("body.left", "body.right"), fingerprint_material="body-a"
    )
    second = derive_actuator_constitution(
        ("body.left", "body.right"), fingerprint_material="body-b"
    )
    assert first.actuator_ids == second.actuator_ids
    assert first.contract_fingerprint != second.contract_fingerprint


@pytest.mark.parametrize(
    "channel_change",
    [
        {"command_min": 0.2, "neutral": 0.5, "command_max": 0.8},
        {"available": False},
        {"slot_id": "effector.left.v2"},
    ],
    ids=["command-range-and-neutral", "availability", "slot-identity"],
)
def test_surface_fingerprint_binds_channel_semantics(channel_change):
    original = derive_actuator_constitution(
        ("effector.left",), fingerprint_material="same-material"
    )
    changed_channel = replace(original.channels[0], **channel_change)
    changed = ActuatorSurface(
        channels=(changed_channel,), fingerprint_material="same-material"
    )

    assert changed.actuator_ids == original.actuator_ids
    assert changed.contract_fingerprint != original.contract_fingerprint


def test_positional_count_construction_is_not_supported():
    with pytest.raises(TypeError):
        derive_actuator_constitution(2)  # type: ignore[arg-type]


@pytest.mark.parametrize("scalar", ["effector.left", b"effector.left"])
def test_scalar_string_is_not_treated_as_sequence_of_effector_ids(scalar):
    with pytest.raises(ValueError, match="sequence, not a scalar string"):
        derive_actuator_constitution(scalar)  # type: ignore[arg-type]


def test_surface_exposes_only_legal_command_contract():
    surface = derive_actuator_constitution(
        ("fixture.effector.left", "fixture.effector.right")
    )
    channel = surface.slots[0]
    assert (channel.command_min, channel.neutral, channel.command_max) == (0.0, 0.0, 1.0)
    assert channel.available is True
    assert not hasattr(channel, "initial_health")
    assert not hasattr(channel, "basal_cost")


def test_constitution_slots_are_immutable_tuple_not_dict():
    constitution = derive_actuator_constitution(
        ("fixture.effector.left", "fixture.effector.right")
    )
    assert isinstance(constitution.slots, tuple)
    with pytest.raises(FrozenInstanceError):
        constitution.channels = ()  # type: ignore[misc]


def test_slot_for_looks_up_by_actuator_id():
    constitution = derive_actuator_constitution(
        ("fixture.effector.left", "fixture.effector.right")
    )
    actuator_id = constitution.actuator_ids[0]
    slot = constitution.slot_for(actuator_id)
    assert slot.actuator_id == actuator_id


def test_slot_for_raises_for_unknown_actuator_id():
    constitution = derive_actuator_constitution(
        ("fixture.effector.left", "fixture.effector.right")
    )
    with pytest.raises(KeyError):
        constitution.slot_for("actuator.does-not-exist")
