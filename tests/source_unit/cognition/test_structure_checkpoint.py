from __future__ import annotations


import pytest

from organism.cognition.structure import StructuralPlasticity


def _restore(payload):
    return StructuralPlasticity.restore_checkpoint(
        payload,
        min_candidate_support=2,
        tentative_lifetime_ticks=4,
        cooldown_ticks=3,
        allowed_node_ids={"a", "b"},
    )


def test_structural_plasticity_checkpoint_round_trips_exactly():
    model = StructuralPlasticity(
        min_candidate_support=2, tentative_lifetime_ticks=4, cooldown_ticks=3
    )
    model._coactivation_counts[("a", "b")] = 3
    model._motor_association_counts[("b", "a")] = 2
    model._cooldown_until["a"] = 8

    restored = _restore(model.export_checkpoint())

    assert restored.export_checkpoint() == model.export_checkpoint()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"coactivation_counts": [], "motor_association_counts": []},
        {
            "coactivation_counts": [
                {"source_id": "a", "target_id": "missing", "count": 1}
            ],
            "motor_association_counts": [],
            "cooldown_until": {},
        },
        {
            "coactivation_counts": [
                {"source_id": "a", "target_id": "b", "count": 1},
                {"source_id": "a", "target_id": "b", "count": 2},
            ],
            "motor_association_counts": [],
            "cooldown_until": {},
        },
    ],
)
def test_structural_plasticity_restore_rejects_missing_or_lossy_payloads(payload):
    with pytest.raises(ValueError):
        _restore(payload)


def test_structural_plasticity_restore_rejects_coerced_identifiers():
    payload = {
        "min_candidate_support": 2,
        "tentative_lifetime_ticks": 4,
        "cooldown_ticks": 3,
        "coactivation_counts": [
            {"source_id": 1, "target_id": "b", "count": 1}
        ],
        "motor_association_counts": [],
        "cooldown_until": {},
    }
    with pytest.raises(ValueError, match="identities"):
        _restore(payload)


def test_structural_plasticity_restore_self_contained_config_and_conflicts():
    model = StructuralPlasticity(
        min_candidate_support=5, tentative_lifetime_ticks=10, cooldown_ticks=6
    )
    model._coactivation_counts[("a", "b")] = 4
    model._cooldown_until["a"] = 12

    chk = model.export_checkpoint()
    # Restore without explicit config arguments
    restored = StructuralPlasticity.restore_checkpoint(chk, allowed_node_ids={"a", "b"})
    assert restored.min_candidate_support == 5
    assert restored.tentative_lifetime_ticks == 10
    assert restored.cooldown_ticks == 6
    assert restored.coactivation_counts == {("a", "b"): 4}
    assert restored.cooldown_until == {"a": 12}
    assert restored.export_checkpoint() == chk

    # Conflict rejection
    with pytest.raises(ValueError, match="min_candidate_support conflicts"):
        StructuralPlasticity.restore_checkpoint(
            chk, min_candidate_support=3, allowed_node_ids={"a", "b"}
        )

    with pytest.raises(ValueError, match="tentative_lifetime_ticks conflicts"):
        StructuralPlasticity.restore_checkpoint(
            chk, tentative_lifetime_ticks=2, allowed_node_ids={"a", "b"}
        )

    with pytest.raises(ValueError, match="cooldown_ticks conflicts"):
        StructuralPlasticity.restore_checkpoint(
            chk, cooldown_ticks=99, allowed_node_ids={"a", "b"}
        )


def test_structural_plasticity_restore_rejects_pre_config_checkpoint_even_with_config():
    pre_config_payload = {
        "coactivation_counts": [],
        "motor_association_counts": [],
        "cooldown_until": {},
    }

    with pytest.raises(ValueError, match="fields do not match the current schema"):
        _restore(pre_config_payload)
