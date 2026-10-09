"""Owner-side binding of low-level controller frames to committed actions."""

from __future__ import annotations

from .action import MotorCommand
from .commitment import ActionCommitment
from .controller import ControllerFrame


def motor_command_from_frame(
    commitment: ActionCommitment,
    frame: ControllerFrame,
    *,
    command_id: str,
    issued_at_tick: int,
) -> MotorCommand:
    """Bind a controller frame to its active Organism action commitment.

    This creates a provenance-bearing command only. It does not validate an
    actuator surface, execute the command, or mutate the commitment/controller.
    The caller supplies the causal tick explicitly; no clock is consulted.
    """
    if not commitment.active:
        raise RuntimeError("cannot issue a motor command for an inactive commitment")
    if frame.controller_id != commitment.controller_id:
        raise ValueError("controller frame does not match the active commitment")
    if frame.commitment_id != commitment.commitment_id:
        raise ValueError("controller frame belongs to a different action commitment")
    if frame.competence_id != commitment.competence_id:
        raise ValueError("controller frame competence does not match the commitment")
    if isinstance(issued_at_tick, bool) or not isinstance(issued_at_tick, int):
        raise ValueError("issued_at_tick must be an integer causal tick")
    if issued_at_tick < commitment.started_tick:
        raise ValueError("issued_at_tick precedes the action commitment")
    if issued_at_tick != frame.generated_at_tick:
        raise ValueError("issued_at_tick must match the controller frame causal tick")

    return MotorCommand(
        command_id=command_id,
        commitment_id=commitment.commitment_id,
        controller_id=frame.controller_id,
        competence_id=frame.competence_id,
        surface_fingerprint=commitment.surface_fingerprint,
        channels=frame.channels,
        issued_at_tick=issued_at_tick,
        embodiment_id=commitment.embodiment_id,
    )


__all__ = ["motor_command_from_frame"]
