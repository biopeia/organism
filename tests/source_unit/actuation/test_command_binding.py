from __future__ import annotations

import pytest

from organism.actuation import (
    ActionCommitment,
    CommitmentStatus,
    ControllerFrame,
    MotorIntent,
    SequenceController,
    motor_command_from_frame,
)


def _commitment(**overrides):
    values = {
        "commitment_id": "commitment.organism.7",
        "proposal_id": "proposal.4",
        "effect_target_id": "effect.forward",
        "competence_id": "competence.walk",
        "started_tick": 12,
        "controller_id": "controller.walk",
        "surface_fingerprint": "surface.sha256:abc",
        "embodiment_id": "embodiment.3",
        "intent_id": "intent.walk",
    }
    values.update(overrides)
    return ActionCommitment(**values)


def test_active_organism_commitment_binds_frame_to_explicit_causal_tick():
    commitment = _commitment()
    controller = SequenceController(
        "controller.walk",
        ((MotorIntent("actuator.left", 0.25), MotorIntent("actuator.right", 0.75)),),
    )
    frame = controller.frame(commitment, tick=13)

    command = motor_command_from_frame(
        commitment,
        frame,
        command_id="command.13.1",
        issued_at_tick=13,
    )

    assert command.command_id == "command.13.1"
    assert command.commitment_id == commitment.commitment_id
    assert command.controller_id == commitment.controller_id
    assert command.competence_id == commitment.competence_id
    assert command.surface_fingerprint == commitment.surface_fingerprint
    assert command.embodiment_id == commitment.embodiment_id
    assert command.issued_at_tick == 13
    assert command.channels == (("actuator.left", 0.25), ("actuator.right", 0.75))
    # The binding creates an Organism command value only; no body execution or
    # commitment lifecycle transition occurs here.
    assert commitment.active
    assert commitment.ended_tick is None


def test_binding_rejects_frame_from_another_controller_or_competence():
    commitment = _commitment()
    foreign_controller = SequenceController(
        "controller.other", ((MotorIntent("actuator.left", 0.5),),)
    )
    with pytest.raises(ValueError, match="controller frame"):
        motor_command_from_frame(
            commitment,
            ControllerFrame(
                "controller.other", commitment.commitment_id,
                commitment.competence_id, (), 12,
            ),
            command_id="command.x",
            issued_at_tick=12,
        )
    with pytest.raises(ValueError, match="controller does not match"):
        foreign_controller.frame(commitment, tick=12)
    assert foreign_controller.step == 0

    mismatched = ControllerFrame(
        "controller.walk", commitment.commitment_id, "competence.other", (), 12
    )
    with pytest.raises(ValueError, match="competence"):
        motor_command_from_frame(
            commitment,
            mismatched,
            command_id="command.x",
            issued_at_tick=12,
        )


def test_binding_rejects_inactive_commitments_and_ticks_before_start():
    inactive = _commitment()
    inactive.terminate(tick=14, status=CommitmentStatus.COMPLETED, reason="done")
    frame = SequenceController("controller.walk", ()).frame(
        _commitment(), tick=14
    )
    with pytest.raises(RuntimeError, match="inactive"):
        motor_command_from_frame(
            inactive, frame, command_id="command.x", issued_at_tick=14
        )

    with pytest.raises(ValueError, match="precedes"):
        motor_command_from_frame(
            _commitment(), frame, command_id="command.x", issued_at_tick=11
        )


def test_binding_requires_explicit_integer_tick():
    commitment = _commitment()
    frame = SequenceController("controller.walk", ()).frame(commitment, tick=12)
    for invalid in (None, True, 12.0):
        with pytest.raises(ValueError, match="integer causal tick"):
            motor_command_from_frame(
                commitment,
                frame,
                command_id="command.x",
                issued_at_tick=invalid,
            )


def test_binding_rejects_frame_from_another_commitment_and_tick():
    commitment = _commitment()
    other_commitment = _commitment(commitment_id="commitment.other")
    foreign_frame = SequenceController("controller.walk", ()).frame(
        other_commitment, tick=12
    )
    with pytest.raises(ValueError, match="different action commitment"):
        motor_command_from_frame(
            commitment,
            foreign_frame,
            command_id="command.x",
            issued_at_tick=12,
        )

    frame = SequenceController("controller.walk", ()).frame(commitment, tick=13)
    for tick in (12, 14):
        with pytest.raises(ValueError, match="must match"):
            motor_command_from_frame(
                commitment,
                frame,
                command_id="command.x",
                issued_at_tick=tick,
            )
