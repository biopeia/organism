"""Low-level controller frames for an existing action commitment."""

from __future__ import annotations

from dataclasses import dataclass

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
    controller_id: str
    sequence: tuple[tuple[MotorIntent, ...], ...]
    step: int = 0

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
