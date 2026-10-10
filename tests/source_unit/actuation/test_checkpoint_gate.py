from __future__ import annotations

from dataclasses import replace

import pytest

from organism.actuation.checkpoint import export_actuation_state, restore_actuation_state
from organism.actuation.context import ActuatorEvidenceContext
from organism.actuation.proposer import ActuatorEvidenceModel
from organism.actuation.surface import ActuatorSurface, derive_actuator_constitution


def _context(
    body_id: str = "body-1",
    episode_id: str = "episode-1",
    morphology_revision: str = "morphology-1",
    *,
    constitution,
    surface_fingerprint: str | None = None,
) -> ActuatorEvidenceContext:
    return ActuatorEvidenceContext(
        body_id=body_id,
        embodiment_episode_id=episode_id,
        morphology_revision=morphology_revision,
        surface_fingerprint=surface_fingerprint or (
            constitution.contract_fingerprint
        ),
    )


def _dormant_candidate_payload(actuator_id: str) -> dict:
    return {
        "actuator_id": actuator_id,
        "activations": 0,
        "cost_evidence": 0.0,
        "probing_state": "dormant",
        "last_seen_tick": 0,
        "natural_promotion_samples": 0,
        "promotion_context_fingerprint": None,
        "context_evidence": {},
        "effect_relations": {},
    }


def _checkpoint(candidates: dict, constitution, organism_id: str = "org-gate") -> dict:
    return {
        "organism_id": organism_id,
        "context": _context(constitution=constitution).to_payload(),
        "candidates": candidates,
    }


def test_restore_raises_when_candidates_key_missing_entirely():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    payload = {}  # no "candidates" key at all
    with pytest.raises(ValueError):
        restore_actuation_state(payload, constitution, organism_id="org-gate", context=_context(constitution=constitution))


def test_restore_accepts_explicit_empty_state_for_a_body_with_no_actuators():
    # An empty "candidates" dict is only valid when the constitution itself
    # defines zero actuators (a genuinely empty body) — set(payload.candidates)
    # must equal set(constitution.actuator_ids) exactly (spec §15/P0.2).
    # export_actuation_state always exports one entry per actuator_id, so an
    # empty candidates dict against a NON-empty constitution is a truncated
    # checkpoint, not a valid "never explored anything" state — see
    # test_restore_raises_when_candidate_set_does_not_match_constitution.
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(0)))
    payload = _checkpoint({}, constitution)
    restored = restore_actuation_state(
        payload, constitution, organism_id="org-gate", context=_context(constitution=constitution)
    )
    assert restored.active_repertoire == ()


def test_restore_raises_when_candidate_set_does_not_match_constitution():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(3)))
    (actuator_id,) = constitution.actuator_ids[:1]
    # Only one of the three actuators the constitution defines is present —
    # a truncated/foreign checkpoint, even though every entry it DOES have
    # is individually well-formed and refers to a real actuator_id.
    payload = _checkpoint({actuator_id: _dormant_candidate_payload(actuator_id)}, constitution)
    with pytest.raises(ValueError):
        restore_actuation_state(payload, constitution, organism_id="org-gate", context=_context(constitution=constitution))


def test_restore_raises_when_candidate_key_does_not_match_its_own_actuator_id_field():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(2)))
    id_a, id_b = constitution.actuator_ids
    # Candidate stored under key id_a, but its own actuator_id field says id_b
    # — both are real, known ids, so the old subset-only check would have
    # passed this silently.
    payload = _checkpoint({
            id_a: _dormant_candidate_payload(id_b),
            id_b: _dormant_candidate_payload(id_b),
        }, constitution)
    with pytest.raises(ValueError):
        restore_actuation_state(payload, constitution, organism_id="org-gate", context=_context(constitution=constitution))


def test_restore_rejects_candidate_actuator_id_not_in_constitution():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    payload = _checkpoint({
            "actuator.not-in-constitution": {
                "actuator_id": "actuator.not-in-constitution",
                "activations": 1,
                "cost_evidence": 0.0,
                "probing_state": "active",
                "last_seen_tick": 0,
                "natural_promotion_samples": 0,
                "promotion_context_fingerprint": None,
                "context_evidence": {},
                "effect_relations": {},
            }
        }, constitution)
    with pytest.raises(ValueError):
        restore_actuation_state(payload, constitution, organism_id="org-gate", context=_context(constitution=constitution))


def test_restore_raises_when_active_candidate_evidence_does_not_satisfy_promotion():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    (actuator_id,) = constitution.actuator_ids
    candidate = _dormant_candidate_payload(actuator_id)
    candidate["probing_state"] = "active"  # claims active with zero recorded evidence
    payload = _checkpoint({actuator_id: candidate}, constitution)
    with pytest.raises(ValueError):
        restore_actuation_state(payload, constitution, organism_id="org-gate", context=_context(constitution=constitution), effect_threshold=0.6)


def test_restore_rejects_candidate_without_current_promotion_marker():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    (actuator_id,) = constitution.actuator_ids
    proposer = ActuatorEvidenceModel(
        constitution,
        organism_id="org-natural-legacy",
        context=_context(constitution=constitution),
        effect_threshold=0.6,
    )
    for tick in range(12):
        activation = 0.2 + 0.05 * tick
        proposer.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation * 0.9,
            tick=tick,
        )
        proposer.consider_natural_evidence(actuator_id, min_samples=12)

    payload = export_actuation_state(proposer)
    candidate = payload["candidates"][actuator_id]
    assert candidate["probing_state"] == "active"

    candidate.pop("natural_promotion_samples", None)

    with pytest.raises(ValueError, match="fields do not match the current schema"):
        restore_actuation_state(
            payload,
            constitution,
            organism_id="org-natural-legacy",
            context=_context(constitution=constitution),
            effect_threshold=0.6,
        )


def test_natural_promotion_threshold_round_trips_explicitly():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    (actuator_id,) = constitution.actuator_ids
    proposer = ActuatorEvidenceModel(
        constitution,
        organism_id="org-natural-new",
        context=_context(constitution=constitution),
        effect_threshold=0.6,
    )
    for tick in range(7):
        activation = 0.1 + 0.1 * tick
        proposer.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation,
            tick=tick,
        )
        proposer.consider_natural_evidence(actuator_id, min_samples=7)

    payload = export_actuation_state(proposer)
    assert payload["candidates"][actuator_id]["natural_promotion_samples"] == 7

    restored = restore_actuation_state(
        payload,
        constitution,
        organism_id="org-natural-new",
        context=_context(constitution=constitution),
        effect_threshold=0.6,
    )
    state = restored.states[0]
    assert state.probing_state == "active"
    assert state.natural_promotion_samples == 7


def test_restore_accepts_naturally_promoted_candidate_after_effect_strength_decays():
    constitution = derive_actuator_constitution(tuple(f"fixture-effector-{index}" for index in range(1)))
    (actuator_id,) = constitution.actuator_ids
    proposer = ActuatorEvidenceModel(
        constitution,
        organism_id="org-natural-decay",
        context=_context(constitution=constitution),
        effect_threshold=0.5,
    )

    # First establish strong natural evidence and promote.
    for tick in range(12):
        activation = 0.2 + 0.05 * tick
        proposer.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation,
            tick=tick,
        )
        proposer.consider_natural_evidence(actuator_id, min_samples=12)

    assert proposer.states[0].probing_state == "active"
    assert proposer.states[0].natural_promotion_samples == 12

    # Then add enough decorrelating experience for the current aggregate to
    # fall below the original promotion threshold. Active status is historical
    # and must remain restorable.
    for tick in range(12, 80):
        activation = 0.2 + 0.01 * (tick % 7)
        delta = 0.3 if tick % 2 else -0.3
        proposer.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=delta,
            tick=tick,
        )

    payload = export_actuation_state(proposer)
    restored = restore_actuation_state(
        payload,
        constitution,
        organism_id="org-natural-decay",
        context=_context(constitution=constitution),
        effect_threshold=0.5,
    )
    assert restored.active_repertoire == (actuator_id,)


def test_restore_rejects_checkpoint_owned_by_a_different_organism():
    constitution = derive_actuator_constitution(("effector.left",))
    model = ActuatorEvidenceModel(constitution, organism_id="organism-a", context=_context(constitution=constitution))
    payload = export_actuation_state(model)

    with pytest.raises(ValueError, match="organism_id"):
        restore_actuation_state(
            payload, constitution, organism_id="organism-b", context=_context(constitution=constitution)
        )


def test_reembodiment_retains_old_evidence_but_requires_new_context_evidence():
    constitution = derive_actuator_constitution(("effector.left",))
    (actuator_id,) = constitution.actuator_ids
    original = ActuatorEvidenceModel(
        constitution, organism_id="organism-a", context=_context(constitution=constitution)
    )
    for tick in range(12):
        activation = 0.1 + 0.06 * tick
        original.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation,
            tick=tick,
        )
    assert original.consider_natural_evidence(actuator_id, min_samples=12)
    payload = export_actuation_state(original)
    prior_context_key = _context(constitution=constitution).fingerprint
    assert prior_context_key in payload["candidates"][actuator_id]["context_evidence"]

    new_context = _context("body-2", "episode-2", constitution=constitution)
    reembodied = restore_actuation_state(
        payload, constitution, organism_id="organism-a", context=new_context
    )
    assert reembodied.active_repertoire == ()
    assert reembodied.states[0].probing_state == "dormant"
    assert reembodied.states[0].activations == 12  # longitudinal evidence was not erased
    assert prior_context_key in reembodied.states[0].context_evidence
    assert not reembodied.consider_natural_evidence(actuator_id, min_samples=12)

    for tick in range(12, 24):
        activation = 0.1 + 0.06 * (tick - 12)
        reembodied.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation,
            tick=tick,
        )
    assert reembodied.consider_natural_evidence(actuator_id, min_samples=12)
    assert reembodied.active_repertoire == (actuator_id,)


@pytest.mark.parametrize(
    "context_change",
    [
        {"surface_changed": True},
        {"body_id": "body-2"},
        {"morphology_revision": "morphology-v2"},
        {"episode_id": "episode-2"},
    ],
    ids=["surface-contract", "body", "morphology-revision", "embodiment-episode"],
)

def test_any_physical_context_change_invalidates_active_applicability(context_change):
    constitution = derive_actuator_constitution(("effector.shared-name",))
    new_constitution = (
        ActuatorSurface(
            channels=(replace(
                constitution.channels[0], command_max=0.75, neutral=0.25
            ),),
            fingerprint_material="",
        )
        if context_change.get("surface_changed")
        else constitution
    )
    new_context = _context(
        constitution=new_constitution,
        **{key: value for key, value in context_change.items() if key != "surface_changed"},
    )
    (actuator_id,) = constitution.actuator_ids
    model = ActuatorEvidenceModel(
        constitution, organism_id="organism-a", context=_context(constitution=constitution)
    )
    for tick in range(12):
        activation = 0.1 + 0.06 * tick
        model.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation,
            tick=tick,
        )
    assert model.consider_natural_evidence(actuator_id, min_samples=12)

    restored = restore_actuation_state(
        export_actuation_state(model),
        new_constitution,
        organism_id="organism-a",
        context=new_context,
    )
    assert restored.active_repertoire == ()
    assert restored.states[0].activations == 12
    assert not restored.consider_natural_evidence(actuator_id, min_samples=12)
