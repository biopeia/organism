"""Low-level controller frames for an existing action commitment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .commitment import ActionCommitment
from .types import MotorIntent


@dataclass(frozen=True, slots=True)
class ControllerFrame:
    """Non-authoritative low-level activations produced by a controller.

    A frame cannot cross the body boundary. ActionDomain must convert it into
    a provenance-bearing MotorCommand under the active ActionCommitment.
    """

    controller_id: str
    commitment_id: str
    competence_id: str | None
    channels: tuple[tuple[str, float], ...]
    generated_at_tick: int

    def __post_init__(self) -> None:
        if not self.controller_id or not self.commitment_id:
            raise ValueError("controller frame provenance must not be empty")
        if isinstance(self.generated_at_tick, bool) or not isinstance(self.generated_at_tick, int):
            raise ValueError("controller frame tick must be an integer causal tick")
        if self.generated_at_tick < 0:
            raise ValueError("controller frame tick must be non-negative")


@dataclass(slots=True)
class SequenceController:
    SCHEMA_VERSION = 1

    controller_id: str
    sequence: tuple[tuple[MotorIntent, ...], ...]
    step: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.controller_id, str) or not self.controller_id:
            raise ValueError("controller_id must be a non-empty string")
        if not isinstance(self.sequence, tuple) or any(
            not isinstance(frame, tuple)
            or any(not isinstance(intent, MotorIntent) for intent in frame)
            for frame in self.sequence
        ):
            raise ValueError("sequence must be a tuple of MotorIntent tuples")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")

    def checkpoint(self) -> dict[str, object]:
        """Return the complete mutable controller continuation state."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "controller_id": self.controller_id,
            "sequence": [
                [
                    {"actuator_id": intent.actuator_id, "activation": intent.activation}
                    for intent in frame
                ]
                for frame in self.sequence
            ],
            "step": self.step,
        }

    @classmethod
    def restore(cls, payload: object) -> "SequenceController":
        """Restore only the exact supported schema; reject partial/unknown state."""
        if not isinstance(payload, Mapping):
            raise ValueError("controller checkpoint must be a mapping")
        expected = {"schema_version", "controller_id", "sequence", "step"}
        if set(payload) != expected:
            raise ValueError("controller checkpoint fields do not match schema")
        version = payload["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != cls.SCHEMA_VERSION:
            raise ValueError("unsupported controller checkpoint schema_version")
        controller_id = payload["controller_id"]
        if not isinstance(controller_id, str) or not controller_id:
            raise ValueError("controller checkpoint controller_id must be non-empty")
        raw_sequence = payload["sequence"]
        if not isinstance(raw_sequence, list):
            raise ValueError("controller checkpoint sequence must be a list")
        sequence: list[tuple[MotorIntent, ...]] = []
        for raw_frame in raw_sequence:
            if not isinstance(raw_frame, list):
                raise ValueError("controller checkpoint frame must be a list")
            frame: list[MotorIntent] = []
            for raw_intent in raw_frame:
                if not isinstance(raw_intent, Mapping) or set(raw_intent) != {
                    "actuator_id", "activation"
                }:
                    raise ValueError("controller checkpoint intent fields do not match schema")
                actuator_id = raw_intent["actuator_id"]
                if not isinstance(actuator_id, str) or not actuator_id:
                    raise ValueError("controller checkpoint actuator_id must be non-empty")
                frame.append(MotorIntent(actuator_id, raw_intent["activation"]))
            sequence.append(tuple(frame))
        step = payload["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("controller checkpoint step must be a non-negative integer")
        return cls(controller_id, tuple(sequence), step)

    def frame(
        self,
        commitment: ActionCommitment,
        *,
        tick: int,
    ) -> ControllerFrame:
        if not commitment.active:
            raise RuntimeError("cannot control an inactive commitment")
        if self.controller_id != commitment.controller_id:
            raise ValueError("controller does not match the active commitment")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("controller tick must be a non-negative integer causal tick")
        if tick < commitment.started_tick:
            raise ValueError("controller tick precedes commitment")
        if not self.sequence:
            channels: tuple[tuple[str, float], ...] = ()
        else:
            intents = self.sequence[min(self.step, len(self.sequence) - 1)]
            channels = tuple((intent.actuator_id, float(intent.activation)) for intent in intents)
            self.step += 1
        return ControllerFrame(
            controller_id=self.controller_id,
            commitment_id=commitment.commitment_id,
            competence_id=commitment.competence_id,
            channels=channels,
            generated_at_tick=tick,
        )


__all__ = ["ControllerFrame", "SequenceController"]
