from __future__ import annotations

from organism.actuation.proposer import ActuatorEvidenceModel
from organism.actuation.context import ActuatorEvidenceContext
from organism.actuation.surface import derive_actuator_constitution


def _constitution(effector_ids: tuple[str, ...] = ("effector.left", "effector.right")):
    return derive_actuator_constitution(effector_ids)


def _context(constitution) -> ActuatorEvidenceContext:
    return ActuatorEvidenceContext(
        "body-1", "episode-1", "morphology-1", constitution.contract_fingerprint
    )


def test_new_proposer_has_all_actuators_dormant():
    constitution = _constitution()
    proposer = ActuatorEvidenceModel(constitution, organism_id="org-1", context=_context(constitution))
    assert proposer.active_repertoire == ()
    assert all(state.probing_state == "dormant" for state in proposer.states)


def test_natural_evidence_can_promote_actuator_from_endogenous_activity():
    """Promotion is exclusively evidence-driven (L6.2): no scheduled
    probing-window apparatus lives in the organism — the actuator becomes
    active purely from activation/percept pairs recorded for whatever
    endogenous reason they occurred."""
    constitution = _constitution(("effector.left",))
    (actuator_id,) = constitution.actuator_ids
    proposer = ActuatorEvidenceModel(
        constitution,
        organism_id="org-natural",
        context=_context(constitution),
        effect_threshold=0.6,
    )

    for tick in range(12):
        activation = 0.2 + 0.05 * tick
        proposer.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation * 0.8,
            tick=tick,
        )
        proposer.consider_natural_evidence(actuator_id, min_samples=12)

    assert actuator_id in proposer.active_repertoire


def test_natural_evidence_does_not_promote_before_minimum_samples():
    constitution = _constitution(("effector.left",))
    (actuator_id,) = constitution.actuator_ids
    proposer = ActuatorEvidenceModel(
        constitution, organism_id="org-natural", context=_context(constitution)
    )

    for tick in range(11):
        activation = 0.1 + 0.06 * tick
        proposer.record_effect(
            actuator_id,
            "percept.x",
            activation=activation,
            delta_percept=activation,
            tick=tick,
        )

    assert proposer.consider_natural_evidence(actuator_id, min_samples=12) is False
    assert proposer.active_repertoire == ()
