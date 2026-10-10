"""Opaque executable actuator contract supplied by the embodiment.

The surface states only which opaque commands are legal.  Response quality,
latency, cost, degradation and functional role are not supplied as organism
knowledge; they must be learned from experienced consequences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from .types import ActuatorId

_SURFACE_SCHEMA = "biopeia-actuator-surface-v1"


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
    contract_fingerprint: str

    def __post_init__(self) -> None:
        ids = [channel.actuator_id for channel in self.channels]
        if len(ids) != len(set(ids)):
            raise ValueError("actuator ids must be unique")
        if not self.contract_fingerprint:
            raise ValueError("contract_fingerprint must not be empty")

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
        # A collection's iteration order is not identity. Canonicalize only the
        # fingerprint input; preserve the supplied order for presentation.
        fingerprint_input = json.dumps(
            [_SURFACE_SCHEMA, fingerprint_material or "", sorted(ids)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fingerprint = sha256(
            fingerprint_input.encode("utf-8")
        ).hexdigest()
        return cls(channels=channels, contract_fingerprint=fingerprint)


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
