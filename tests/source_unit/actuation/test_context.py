from __future__ import annotations

import pytest

from organism.actuation.context import ActuatorEvidenceContext
from organism.actuation.proposer import ActuatorEvidenceModel
from organism.actuation.surface import derive_actuator_constitution


def test_context_requires_explicit_nonempty_identity_and_contract_fields():
    with pytest.raises(ValueError):
        ActuatorEvidenceContext(
            body_id="body-1",
            embodiment_episode_id="episode-1",
            morphology_revision="",
            surface_fingerprint="surface-1",
        )


def test_context_fingerprint_is_stable_and_binds_every_identity_field():
    context = ActuatorEvidenceContext("body-1", "episode-1", "morph-1", "surface-1")
    assert context.fingerprint == ActuatorEvidenceContext(
        "body-1", "episode-1", "morph-1", "surface-1"
    ).fingerprint
    assert context.fingerprint != ActuatorEvidenceContext(
        "body-2", "episode-1", "morph-1", "surface-1"
    ).fingerprint
    assert context.fingerprint != ActuatorEvidenceContext(
        "body-1", "episode-2", "morph-1", "surface-1"
    ).fingerprint
    assert context.fingerprint != ActuatorEvidenceContext(
        "body-1", "episode-1", "morph-2", "surface-1"
    ).fingerprint
    assert context.fingerprint != ActuatorEvidenceContext(
        "body-1", "episode-1", "morph-1", "surface-2"
    ).fingerprint


def test_context_payload_is_closed_and_round_trips():
    context = ActuatorEvidenceContext("body-1", "episode-1", "morph-1", "surface-1")
    assert ActuatorEvidenceContext.from_payload(context.to_payload()) == context
    with pytest.raises(ValueError, match="schema"):
        ActuatorEvidenceContext.from_payload({**context.to_payload(), "extra": "x"})


def test_evidence_model_rejects_context_for_a_different_surface_contract():
    constitution = derive_actuator_constitution(
        ("effector.left",), fingerprint_material="contract-a"
    )
    wrong_context = ActuatorEvidenceContext(
        "body-1", "episode-1", "morphology-1", "different-contract"
    )

    with pytest.raises(ValueError, match="active ActuatorConstitution"):
        ActuatorEvidenceModel(
            constitution, organism_id="organism-1", context=wrong_context
        )
