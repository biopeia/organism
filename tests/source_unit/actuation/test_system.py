from __future__ import annotations

import pytest

from organism.actuation.action import ActionSource, MotorCommand
from organism.actuation.commitment import ActionCommitment, CommitmentStatus
from organism.actuation.surface import derive_actuator_constitution
from organism.actuation.system import _ActuatorSystem


def _command(surface, actuator_id: str, activation: float) -> MotorCommand:
    return MotorCommand.from_mapping(
        command_id="command.test",
        commitment_id="commitment.test",
        controller_id="controller.test",
        competence_id=None,
        surface_fingerprint=surface.contract_fingerprint,
        channels={actuator_id: activation},
        issued_at_tick=1,
        embodiment_id="embodiment.test",
    )


def _commitment(surface, *, commitment_id: str = "commitment.test") -> ActionCommitment:
    return ActionCommitment(
        commitment_id=commitment_id,
        proposal_id="proposal.test",
        effect_target_id=None,
        competence_id=None,
        started_tick=0,
        controller_id="controller.test",
        surface_fingerprint=surface.contract_fingerprint,
        proposal_source=ActionSource.EXPLORATION,
        embodiment_id="embodiment.test",
        maximum_duration=1,
    )


def test_system_hands_committed_command_to_body_boundary_without_interpreting_health():
    surface = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    actuator_id = surface.actuator_ids[0]
    actuations = _ActuatorSystem()._execute_command(
        _command(surface, actuator_id, 0.8),
        surface,
        commitment=_commitment(surface),
    )
    assert len(actuations) == 1
    assert actuations[0].requested == 0.8
    assert actuations[0].delivered == 0.8


def test_system_rejects_channel_outside_surface():
    surface = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    with pytest.raises(KeyError):
        _ActuatorSystem()._execute_command(
            _command(surface, "actuator.other", 0.5),
            surface,
            commitment=_commitment(surface),
        )


def test_system_rejects_command_for_another_surface():
    surface = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)), fingerprint_material="surface.a")
    foreign = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)), fingerprint_material="surface.b")
    with pytest.raises(RuntimeError, match="different actuator surface"):
        _ActuatorSystem()._execute_command(
            _command(foreign, foreign.actuator_ids[0], 0.5),
            surface,
            commitment=_commitment(foreign),
        )


def test_system_rejects_missing_inactive_or_mismatched_commitment():
    surface = derive_actuator_constitution(("effector.a",))
    command = _command(surface, "effector.a", 0.5)
    inactive = _commitment(surface)
    inactive.terminate(tick=1, status=CommitmentStatus.COMPLETED, reason="fixture")
    with pytest.raises(RuntimeError, match="not active"):
        _ActuatorSystem()._execute_command(command, surface, commitment=inactive)

    foreign = _commitment(surface, commitment_id="commitment.other")
    with pytest.raises(RuntimeError, match="does not match"):
        _ActuatorSystem()._execute_command(command, surface, commitment=foreign)


def test_surface_is_command_contract_not_body_model():
    channel = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1))).slots[0]
    assert hasattr(channel, "command_min")
    assert hasattr(channel, "command_max")
    assert hasattr(channel, "neutral")
    assert not hasattr(channel, "initial_health")
    assert not hasattr(channel, "basal_cost")
    assert not hasattr(channel, "execution_threshold")
