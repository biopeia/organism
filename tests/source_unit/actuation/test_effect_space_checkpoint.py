from __future__ import annotations

from copy import deepcopy
import json

import pytest

from organism.actuation.effects import EffectAtom, EffectSpace


def test_effect_space_checkpoint_restores_and_continues() -> None:
    space = EffectSpace(max_effects=4)
    space.observe({"signal.a": 0.5})
    space.observe({"signal.a": 0.5})
    space.register_footprint(
        "effect.entity.body.part",
        (EffectAtom("part.segment", 1, 2),),
        support=2,
    )

    persisted = json.loads(json.dumps(space.checkpoint()))
    restored = EffectSpace.restore(persisted)
    original = space.observe({"signal.a": 0.5})
    continued = restored.observe({"signal.a": 0.5})

    assert continued == original
    assert restored.checkpoint() == space.checkpoint()


def test_effect_space_restore_requires_exact_top_level_schema() -> None:
    valid = EffectSpace().checkpoint()
    for key in tuple(valid):
        missing = deepcopy(valid)
        del missing[key]
        with pytest.raises(ValueError):
            EffectSpace.restore(missing)

    with pytest.raises(ValueError):
        EffectSpace.restore({**valid, "future": None})
    for invalid in (None, [], "checkpoint"):
        with pytest.raises(ValueError):
            EffectSpace.restore(invalid)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unsupported"):
        EffectSpace.restore({**valid, "schema_version": 2})


@pytest.mark.parametrize("value", [True, 1.5, "4", 0, -1])
def test_effect_space_restore_rejects_invalid_capacity(value: object) -> None:
    valid = EffectSpace().checkpoint()
    valid["max_effects"] = value
    with pytest.raises(ValueError):
        EffectSpace.restore(valid)


def test_effect_space_restore_rejects_malformed_or_excess_support() -> None:
    valid = EffectSpace(max_effects=1).checkpoint()
    for support in (None, {}, [None], [{"signature": [], "count": 1}],
                    [{"signature": [["joint.knee", 1]], "count": 1}],
                    [{"signature": [["signal.a", 0]], "count": 1}],
                    [{"signature": [["signal.a", 8]], "count": 1}],
                    [{"signature": [["signal.a", True]], "count": 1}],
                    [{"signature": [["signal.a", 1], ["signal.a", 2]], "count": 1}],
                    [{"signature": [["signal.a", 1]], "count": True}],
                    [{"signature": [["signal.a", 1]], "count": 0}],
                    [{"signature": [["signal.a", 1]]}],
                    [{"signature": [["signal.a", 1]], "count": 1,
                      "unknown": 1}],):
        with pytest.raises(ValueError):
            EffectSpace.restore({**valid, "support": support})

    entry = {"signature": [["signal.a", 1]], "count": 1}
    with pytest.raises(ValueError):
        EffectSpace.restore({**valid, "support": [entry, entry]})
    over_capacity = EffectSpace(max_effects=2)
    over_capacity.observe({"signal.a": 0.5})
    over_capacity.observe({"signal.b": 0.5})
    checkpoint = over_capacity.checkpoint()
    checkpoint["max_effects"] = 1
    with pytest.raises(ValueError):
        EffectSpace.restore(checkpoint)


def test_effect_space_restore_rejects_malformed_or_duplicate_footprints() -> None:
    valid = EffectSpace().checkpoint()
    entry = {"effect_id": "effect.entity.body.part", "support": 1,
             "atoms": [["signal.a", 1, 1]]}
    for footprints in (None, {}, [None], [{"effect_id": "effect.entity.x"}],
                       [{**entry, "effect_id": 2}],
                       [{**entry, "support": True}],
                       [{**entry, "support": 0}],
                       [{**entry, "atoms": None}],
                       [{**entry, "atoms": []}],
                       [{**entry, "atoms": [["joint.knee", 1, 1]]}],
                       [{**entry, "atoms": [["signal.a", 0, 1]]}],
                       [{**entry, "atoms": [["signal.a", 1, 4]]}],
                       [{**entry, "atoms": [["signal.a", True, 1]]}],
                       [{**entry, "atoms": [["signal.a", 1, 1, 2]]}],
                       [{**entry, "unknown": 1}]):
        with pytest.raises(ValueError):
            EffectSpace.restore({**valid, "footprints": footprints})

    with pytest.raises(ValueError):
        EffectSpace.restore({**valid, "footprints": [entry, entry]})


def test_effect_space_restore_rejects_missing_capacity_pressure() -> None:
    valid = EffectSpace().checkpoint()
    for value in (
        None,
        {**valid["capacity_pressure"], "capacity": 0},
        {**valid["capacity_pressure"], "unknown": 1},
    ):
        with pytest.raises(ValueError):
            EffectSpace.restore({**valid, "capacity_pressure": value})

