from __future__ import annotations

from copy import deepcopy

import pytest

from organism.actuation.composition import CompositionEngine


def test_composition_checkpoint_restores_and_continues_evidence() -> None:
    engine = CompositionEngine(max_relations=2)
    engine.observe("competence.a", "competence.b", "effect.x", success=True)
    engine.observe("competence.a", "competence.b", "effect.x", success=False)

    restored = CompositionEngine.restore(engine.checkpoint())
    evidence = restored.observe(
        "competence.a", "competence.b", "effect.x", success=True
    )

    assert evidence.support == 2
    assert evidence.failures == 1
    assert restored.checkpoint() == {
        "schema_version": 1,
        "max_relations": 2,
        "sequential": [
            {
                "first_competence_id": "competence.a",
                "second_competence_id": "competence.b",
                "effect_id": "effect.x",
                "support": 2,
                "failures": 1,
            }
        ],
    }


@pytest.mark.parametrize("payload", [None, [], "not-a-checkpoint"])
def test_composition_restore_rejects_non_mapping_payload(payload: object) -> None:
    with pytest.raises(ValueError):
        CompositionEngine.restore(payload)  # type: ignore[arg-type]


def test_composition_restore_rejects_missing_or_unknown_top_level_fields() -> None:
    valid = CompositionEngine().checkpoint()

    for key in ("schema_version", "max_relations", "sequential"):
        missing = deepcopy(valid)
        del missing[key]
        with pytest.raises(ValueError):
            CompositionEngine.restore(missing)

    unknown = {**valid, "future": []}
    with pytest.raises(ValueError):
        CompositionEngine.restore(unknown)


@pytest.mark.parametrize("max_relations", [True, 0, -1, 1.5, "2"])
def test_composition_restore_rejects_invalid_capacity(max_relations: object) -> None:
    payload = CompositionEngine().checkpoint()
    payload["max_relations"] = max_relations

    with pytest.raises(ValueError):
        CompositionEngine.restore(payload)


def test_composition_restore_rejects_missing_malformed_and_excess_relations() -> None:
    valid = CompositionEngine(max_relations=1).checkpoint()

    missing_list = {**valid}
    del missing_list["sequential"]
    with pytest.raises(ValueError):
        CompositionEngine.restore(missing_list)

    for sequential in (None, {}, [None], [valid], [{"first_competence_id": "a"}]):
        malformed = {**valid, "sequential": sequential}
        with pytest.raises(ValueError):
            CompositionEngine.restore(malformed)

    too_many = CompositionEngine(max_relations=2)
    too_many.observe("competence.a", "competence.b", "effect.x", success=True)
    too_many.observe("competence.a", "competence.c", "effect.y", success=True)
    too_many.observe("competence.a", "competence.d", "effect.z", success=True)
    # A normal live engine prunes to its own configured limit; represent a
    # corrupted payload whose entries exceed that limit explicitly.
    payload = too_many.checkpoint()
    payload["max_relations"] = 1
    with pytest.raises(ValueError):
        CompositionEngine.restore(payload)


def test_composition_restore_rejects_coercion_duplicates_and_invalid_counts() -> None:
    payload = CompositionEngine().checkpoint()
    base = {
        "first_competence_id": "competence.a",
        "second_competence_id": "competence.b",
        "effect_id": "effect.x",
        "support": 1,
        "failures": 0,
    }

    for field, value in (
        ("first_competence_id", 12),
        ("second_competence_id", ""),
        ("effect_id", "not-an-effect"),
        ("support", True),
        ("support", "1"),
        ("failures", -1),
    ):
        invalid_item = {**base, field: value}
        with pytest.raises(ValueError):
            CompositionEngine.restore({**payload, "sequential": [invalid_item]})

    with pytest.raises(ValueError):
        CompositionEngine.restore({**payload, "sequential": [base, base]})

    with pytest.raises(ValueError):
        CompositionEngine.restore(
            {**payload, "sequential": [{**base, "extra": "ignored before"}]}
        )
