from __future__ import annotations

from copy import deepcopy

import pytest

from organism.agency.intention import ActionIntent, AdmissionRoute, IntentStatus


def _intent() -> ActionIntent:
    return ActionIntent(
        intent_id="intent.test",
        competence_id="competence.test",
        anticipated_effect_id=None,
        context_ref=None,
        embodiment_id=None,
        origin_refs=("memory.test",),
        prediction_ref=None,
        confidence=0.5,
        epistemic_relevance=0.25,
        homeostatic_relevance=0.75,
        supporting_affordance_id=None,
        created_tick=4,
        activated_tick=None,
        last_progress_tick=4,
        status=IntentStatus.PENDING,
        termination_reason=None,
        admission=AdmissionRoute.PROSPECTIVE,
    )


def test_restore_roundtrips_exact_writer_payload_and_nullable_fields() -> None:
    payload = _intent().checkpoint()

    restored = ActionIntent.restore(payload)

    assert restored.checkpoint() == payload
    assert restored.anticipated_effect_id is None
    assert restored.activated_tick is None
    assert restored.termination_reason is None
    assert restored.admission is AdmissionRoute.PROSPECTIVE


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda data: data.pop("confidence"), ValueError, "missing or unknown fields"),
        (lambda data: data.__setitem__("legacy", True), ValueError, "missing or unknown fields"),
        (lambda data: data.__setitem__("intent_id", 9), TypeError, "intent_id must be a string"),
        (lambda data: data.__setitem__("competence_id", None), TypeError, "competence_id must be a string"),
        (lambda data: data.__setitem__("anticipated_effect_id", 7), TypeError, "anticipated_effect_id must be a string or null"),
        (lambda data: data.__setitem__("context_ref", False), TypeError, "context_ref must be a string or null"),
        (lambda data: data.__setitem__("embodiment_id", []), TypeError, "embodiment_id must be a string or null"),
        (lambda data: data.__setitem__("origin_refs", ("memory.test",)), TypeError, "origin_refs must be a list of strings"),
        (lambda data: data.__setitem__("origin_refs", ["memory.test", 3]), TypeError, "origin_refs must be a list of strings"),
        (lambda data: data.__setitem__("prediction_ref", 3), TypeError, "prediction_ref must be a string or null"),
        (lambda data: data.__setitem__("confidence", "0.5"), TypeError, "confidence must be a float"),
        (lambda data: data.__setitem__("epistemic_relevance", 1), TypeError, "epistemic_relevance must be a float"),
        (lambda data: data.__setitem__("homeostatic_relevance", True), TypeError, "homeostatic_relevance must be a float"),
        (lambda data: data.__setitem__("supporting_affordance_id", 2), TypeError, "supporting_affordance_id must be a string or null"),
        (lambda data: data.__setitem__("created_tick", "4"), TypeError, "created_tick must be an integer"),
        (lambda data: data.__setitem__("created_tick", True), TypeError, "created_tick must be an integer"),
        (lambda data: data.__setitem__("activated_tick", "5"), TypeError, "activated_tick must be an integer or null"),
        (lambda data: data.__setitem__("last_progress_tick", 4.0), TypeError, "last_progress_tick must be an integer"),
        (lambda data: data.__setitem__("status", 1), TypeError, "status must be a string"),
        (lambda data: data.__setitem__("termination_reason", False), TypeError, "termination_reason must be a string or null"),
        (lambda data: data.__setitem__("admission", None), TypeError, "admission must be a string"),
    ],
)
def test_restore_rejects_missing_unknown_or_wrong_typed_fields(mutation, error, message) -> None:
    payload = deepcopy(_intent().checkpoint())
    mutation(payload)

    with pytest.raises(error, match=message):
        ActionIntent.restore(payload)


def test_restore_rejects_non_mapping_payload() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        ActionIntent.restore([])  # type: ignore[arg-type]


def test_restore_rejects_non_string_field_names() -> None:
    payload = _intent().checkpoint()
    payload[1] = "unexpected"  # type: ignore[index]

    with pytest.raises(TypeError, match="field names must be strings"):
        ActionIntent.restore(payload)


def test_restore_rejects_invalid_enum_and_non_finite_unit_value() -> None:
    payload = _intent().checkpoint()
    payload["status"] = "not-a-status"
    with pytest.raises(ValueError):
        ActionIntent.restore(payload)

    payload = _intent().checkpoint()
    payload["confidence"] = float("nan")
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        ActionIntent.restore(payload)
