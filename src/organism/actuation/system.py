"""Canonical body-boundary execution for Sensorimotor v2."""

from __future__ import annotations

from .action import MotorCommand
from .commitment import ActionCommitment
from .surface import ActuatorSurface
from .types import Actuation


class _ActuatorSystem:
    """Private low-level executor called only by the registered ActionDomain."""

    def _execute_command(
        self,
        command: MotorCommand,
        surface: ActuatorSurface,
        *,
        commitment: ActionCommitment,
    ) -> tuple[Actuation, ...]:
        if not commitment.active:
            raise RuntimeError("motor command commitment is not active")
        if (
            command.commitment_id != commitment.commitment_id
            or command.controller_id != commitment.controller_id
            or command.competence_id != commitment.competence_id
            or command.surface_fingerprint != commitment.surface_fingerprint
            or command.embodiment_id != commitment.embodiment_id
        ):
            raise RuntimeError("motor command provenance does not match its commitment")
        if command.surface_fingerprint != surface.contract_fingerprint:
            raise RuntimeError("motor command targets a different actuator surface")
        delivered: list[Actuation] = []
        for actuator_id, activation in command.channels:
            value = surface.validate(actuator_id, activation)
            delivered.append(
                Actuation(
                    actuator_id=actuator_id,
                    requested=activation,
                    delivered=value,
                )
            )
        return tuple(delivered)
