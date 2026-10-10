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
        "coactivation_counts": [
            {"source_id": 1, "target_id": "b", "count": 1}
        ],
        "motor_association_counts": [],
        "cooldown_until": {},
    }
    with pytest.raises(ValueError, match="identities"):
        _restore(payload)
