"""Strict continuation tests for the causal-evidence ledger codec."""

from __future__ import annotations

from copy import deepcopy

import pytest

from organism.actuation.evidence import CausalEvidenceLedger, SensorimotorTransition


def _ledger() -> CausalEvidenceLedger:
    ledger = CausalEvidenceLedger(capacity=2, passive_capacity=1)
    ledger.observe(
        SensorimotorTransition(
            transition_id="transition.1",
            tick_start=1,
            tick_end=2,
            context_ref="context.1",
            commitment_id="commitment.1",
            controller_id="controller.1",
            competence_id="competence.1",
            attempt_id="attempt.1",
            intervention_signature_id="signature.1",
            state_before_ref="state.before",
            motor_command_ref="command.1",
            actuation_ref="actuation.1",
            prediction_ref=None,
            state_after_ref="state.after",
            observed_effect_id="effect.change",
            observed_effect_atoms=("effect.atom.change",),
        )
    )
    ledger.observe_passive_window(
        tick_start=2,
        tick_end=3,
        context_ref=None,
        prior_state_ref="state.after",
        resulting_state_ref="state.passive",
        effect_id=None,
    )
    return ledger


def test_writer_checkpoint_round_trips_exactly_through_json_shape():
    import json

    payload = json.loads(json.dumps(_ledger().checkpoint()))
    restored = CausalEvidenceLedger.restore(payload)
    assert restored.checkpoint() == payload
    assert len(restored.intervention_evidence) == 1
    assert len(restored.passive_evidence) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("capacity"),
        lambda payload: payload.update(extra=True),
        lambda payload: payload.update(schema_version=True),
        lambda payload: payload.update(capacity="2"),
        lambda payload: payload.update(passive_capacity=None),
        lambda payload: payload.update(evidence={}),
        lambda payload: payload["evidence"][0].pop("effect_atoms"),
        lambda payload: payload["evidence"][0].update(extra=True),
        lambda payload: payload["evidence"][0].update(observation_tick=True),
        lambda payload: payload["evidence"][0].update(observation_tick="2"),
        lambda payload: payload["evidence"][0].update(attempt_id=1),
        lambda payload: payload["evidence"][0].update(window_start_tick=False),
        lambda payload: payload["evidence"][0].update(window_start_tick=3),
        lambda payload: payload["evidence"][0].update(effect_atoms="effect.atom.change"),
        lambda payload: payload["evidence"][0].update(effect_atoms=[1]),
        lambda payload: payload["evidence"][0].update(effect_atoms=["effect.atom.z", "effect.atom.a"]),
        lambda payload: payload["evidence"][0].update(action_ref=None),
        lambda payload: payload["evidence"][0].update(evidence_id=""),
        lambda payload: payload["evidence"][0].update(attempt_id=None),
        lambda payload: payload["evidence"][0].update(transition_id=None),
        lambda payload: payload["passive_evidence"][0].update(prior_state_ref=None),
        lambda payload: payload["passive_evidence"][0].update(resulting_state_ref=""),
        lambda payload: payload["passive_evidence"][0].update(window_start_tick=None),
    ],
)
def test_rejects_missing_unknown_malformed_or_semantically_inconsistent_payload(mutation):
    payload = deepcopy(_ledger().checkpoint())
    mutation(payload)
    with pytest.raises(ValueError):
        CausalEvidenceLedger.restore(payload)


def test_rejects_over_capacity_checkpoint_instead_of_truncating_history():
    payload = _ledger().checkpoint()
    payload["evidence"].append(deepcopy(payload["evidence"][0]))
    payload["evidence"].append(deepcopy(payload["evidence"][0]))
    with pytest.raises(ValueError, match="exceeds declared capacity"):
        CausalEvidenceLedger.restore(payload)


def test_rejects_over_capacity_passive_checkpoint():
    payload = _ledger().checkpoint()
    payload["passive_evidence"].append(deepcopy(payload["passive_evidence"][0]))
    with pytest.raises(ValueError, match="exceeds declared capacity"):
        CausalEvidenceLedger.restore(payload)


def test_rejects_evidence_in_wrong_window_class():
    payload = _ledger().checkpoint()
    payload["evidence"], payload["passive_evidence"] = (
        payload["passive_evidence"],
        payload["evidence"],
    )
    with pytest.raises(ValueError, match="wrong window class"):
        CausalEvidenceLedger.restore(payload)


def test_rejects_empty_passive_window():
    payload = _ledger().checkpoint()
    payload["passive_evidence"][0]["window_start_tick"] = 3
    with pytest.raises(ValueError, match="window bounds"):
        CausalEvidenceLedger.restore(payload)


@pytest.mark.parametrize(
    ("prior_state_ref", "resulting_state_ref"),
    [("", "state.after"), ("state.before", "")],
)
def test_writer_rejects_empty_passive_state_references(prior_state_ref, resulting_state_ref):
    ledger = CausalEvidenceLedger(capacity=2, passive_capacity=1)
    with pytest.raises(ValueError, match="state reference must not be empty"):
        ledger.observe_passive_window(
            tick_start=1,
            tick_end=2,
            context_ref=None,
            prior_state_ref=prior_state_ref,
            resulting_state_ref=resulting_state_ref,
            effect_id=None,
        )


@pytest.mark.parametrize("tick_start, tick_end", [(1.0, 2), (True, 2), (1, 2.0)])
def test_passive_writer_rejects_non_integer_ticks(tick_start, tick_end):
    ledger = CausalEvidenceLedger(capacity=2, passive_capacity=1)
    with pytest.raises(TypeError, match="ticks must be integers"):
        ledger.observe_passive_window(
            tick_start=tick_start,
            tick_end=tick_end,
            context_ref=None,
            prior_state_ref="state.before",
            resulting_state_ref="state.after",
            effect_id=None,
        )


@pytest.mark.parametrize("tick_start, tick_end", [(1.0, 2), (True, 2), (1, 2.0)])
def test_action_transition_writer_rejects_non_integer_ticks(tick_start, tick_end):
    with pytest.raises(TypeError, match="ticks must be integers"):
        SensorimotorTransition(
            transition_id="transition.bad-ticks",
            tick_start=tick_start,
            tick_end=tick_end,
            context_ref="context.1",
            commitment_id="commitment.1",
            controller_id="controller.1",
            competence_id="competence.1",
            attempt_id="attempt.1",
            intervention_signature_id="signature.1",
            state_before_ref="state.before",
            motor_command_ref="command.1",
            actuation_ref="actuation.1",
            prediction_ref=None,
            state_after_ref="state.after",
        )


@pytest.mark.parametrize(
    "field",
    ["context_ref", "motor_command_ref", "attempt_id", "prediction_ref"],
)
def test_action_transition_writer_rejects_non_string_references(field):
    values = {
        "transition_id": "transition.1",
        "tick_start": 1,
        "tick_end": 2,
        "context_ref": "context.1",
        "commitment_id": "commitment.1",
        "controller_id": "controller.1",
        "competence_id": "competence.1",
        "attempt_id": "attempt.1",
        "intervention_signature_id": "signature.1",
        "state_before_ref": "state.before",
        "motor_command_ref": "command.1",
        "actuation_ref": "actuation.1",
        "prediction_ref": None,
        "state_after_ref": "state.after",
    }
    values[field] = 7
    with pytest.raises((TypeError, ValueError)):
        SensorimotorTransition(**values)


@pytest.mark.parametrize("field", ["context_ref", "effect_id"])
def test_passive_writer_rejects_non_string_optional_references(field):
    values = {
        "tick_start": 1,
        "tick_end": 2,
        "context_ref": None,
        "prior_state_ref": "state.before",
        "resulting_state_ref": "state.after",
        "effect_id": None,
    }
    values[field] = 7
    with pytest.raises(TypeError):
        CausalEvidenceLedger(capacity=2, passive_capacity=1).observe_passive_window(
            **values
        )


def test_rejects_non_object_checkpoint():
    with pytest.raises(ValueError, match="must be an object"):
        CausalEvidenceLedger.restore(None)  # type: ignore[arg-type]
