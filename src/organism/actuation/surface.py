"""Opaque executable actuator contract supplied by the embodiment.

The surface states only which opaque commands are legal.  Response quality,
latency, cost, degradation and functional role are not supplied as organism
knowledge; they must be learned from experienced consequences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Sequence

from .types import ActuatorId

_SURFACE_SCHEMA = "biopeia-actuator-surface-v2"


@dataclass(frozen=True, slots=True)
class ActuatorChannel:
    slot_id: str
    actuator_id: ActuatorId
    command_min: float = 0.0
    command_max: float = 1.0
    neutral: float = 0.0
    available: bool = True

    def __post_init__(self) -> None:
        if not self.slot_id or not self.actuator_id:
            raise ValueError("actuator channel ids must be non-empty")
        minimum = float(self.command_min)
        maximum = float(self.command_max)
        neutral = float(self.neutral)
        if not 0.0 <= minimum <= neutral <= maximum <= 1.0:
            raise ValueError(
                "actuator command contract must satisfy 0 <= min <= neutral <= max <= 1"
            )
        if not isinstance(self.available, bool):
            raise ValueError("available must be boolean")


MotorSlot = ActuatorChannel


@dataclass(frozen=True, slots=True)
class ActuatorSurface:
    channels: tuple[ActuatorChannel, ...]
    fingerprint_material: str = ""
    contract_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.channels, tuple) or any(
            not isinstance(channel, ActuatorChannel) for channel in self.channels
        ):
            raise ValueError("channels must be an immutable tuple of ActuatorChannel values")
        if not isinstance(self.fingerprint_material, str):
            raise ValueError("fingerprint_material must be a string")
        ids = [channel.actuator_id for channel in self.channels]
        if len(ids) != len(set(ids)):
            raise ValueError("actuator ids must be unique")
        fingerprint_input = json.dumps(
            [
                _SURFACE_SCHEMA,
                self.fingerprint_material,
                [
                    {
                        "actuator_id": channel.actuator_id,
                        "slot_id": channel.slot_id,
                        "command_min": channel.command_min,
                        "command_max": channel.command_max,
                        "neutral": channel.neutral,
                        "available": channel.available,
                    }
                    for channel in sorted(self.channels, key=lambda item: item.actuator_id)
                ],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fingerprint = sha256(fingerprint_input.encode("utf-8")).hexdigest()
        object.__setattr__(self, "contract_fingerprint", fingerprint)

    @property
    def slots(self) -> tuple[ActuatorChannel, ...]:
        return self.channels

    @property
    def actuator_ids(self) -> tuple[ActuatorId, ...]:
        return tuple(channel.actuator_id for channel in self.channels)

    def slot_for(self, actuator_id: ActuatorId) -> ActuatorChannel:
        for channel in self.channels:
            if channel.actuator_id == actuator_id:
                return channel
        raise KeyError(f"unknown actuator_id {actuator_id!r}")

    def validate(self, actuator_id: ActuatorId, activation: float) -> float:
        channel = self.slot_for(actuator_id)
        if not channel.available:
            raise RuntimeError(f"actuator channel {actuator_id!r} is unavailable")
        value = float(activation)
        if not channel.command_min <= value <= channel.command_max:
            raise ValueError("motor command outside actuator contract")
        return value

    @classmethod
    def from_effector_ids(
        cls,
        effector_ids: Sequence[str],
        *,
        fingerprint_material: str | None = None,
    ) -> "ActuatorSurface":
        if isinstance(effector_ids, (str, bytes)):
            raise ValueError("effector ids must be a sequence, not a scalar string")
        ids = tuple(effector_ids)
        if len(ids) > 256 or any(
            not isinstance(value, str) or not value.strip() for value in ids
        ):
            raise ValueError("effector ids must be non-empty and bounded to 256")
        if len(ids) != len(set(ids)):
            raise ValueError("effector ids must be unique")
        channels = tuple(
            ActuatorChannel(slot_id=f"effector.{effector_id}", actuator_id=effector_id)
            for effector_id in ids
        )
        # A collection's iteration order is not identity. The constructor
        # fingerprints every semantic channel field in canonical ID order,
        # while preserving the supplied channel order for presentation.
        return cls(channels=channels, fingerprint_material=fingerprint_material or "")


ActuatorConstitution = ActuatorSurface


def derive_actuator_constitution(
    effector_ids: Sequence[str],
    *,
    fingerprint_material: str | None = None,
) -> ActuatorConstitution:
    """Build a command surface keyed by stable, owner-supplied effector IDs."""
    return ActuatorSurface.from_effector_ids(
        effector_ids,
        fingerprint_material=fingerprint_material,
    )


__all__ = [
    "ActuatorChannel",
    "ActuatorConstitution",
    "ActuatorSurface",
    "MotorSlot",
    "derive_actuator_constitution",
]
