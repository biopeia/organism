from __future__ import annotations

import copy

import pytest

from organism.genetics import (
    EpigeneticMark,
    GermlineState,
    GeneExpressionState,
    ExpressionRegulator,
    RegulatorySignals,
    create_standard_genome,
    export_expression,
    export_germline,
    restore_expression,
    restore_germline,
)
from organism.genetics.expression import restore_expression_state


def _genome():
    return create_standard_genome("germline-checkpoint-test")


def test_germline_checkpoint_round_trip_preserves_derived_state():
    genome = _genome()
    inherited = EpigeneticMark(
        "plasticity.learning_rate.baseline", delta=0.01, strength=0.5
    )
    acquired = EpigeneticMark(
        "structure.growth_threshold.baseline", delta=0.02, strength=0.5
    )
    state = GermlineState.from_genome(
        genome,
        inherited_marks=(inherited,),
        acquired_capture_enabled=True,
    )
    state.acquired_marks[acquired.locus] = acquired
    payload = export_germline(state)

    restored = restore_germline(payload, genome)

    assert export_germline(restored) == payload


def test_germline_restore_rejects_omitted_persisted_capture_streaks():
    genome = _genome()
    payload = export_germline(GermlineState.from_genome(genome))
    del payload["capture_streaks"]

    with pytest.raises(ValueError, match="keys mismatch"):
        restore_germline(payload, genome)


@pytest.mark.parametrize("field", ["inherited_marks", "acquired_marks"])
def test_germline_restore_rejects_non_object_mark_entry(field):
    genome = _genome()
    payload = export_germline(GermlineState.from_genome(genome))
    payload[field] = ["junk"]

    with pytest.raises(ValueError, match="entries must be objects"):
        restore_germline(payload, genome)


@pytest.mark.parametrize("field", ["inherited_marks", "acquired_marks"])
def test_germline_restore_rejects_duplicate_mark_locus(field):
    genome = _genome()
    mark = EpigeneticMark("plasticity.learning_rate.baseline", delta=0.01)
    state = GermlineState.from_genome(
        genome,
        inherited_marks=(mark,) if field == "inherited_marks" else (),
        acquired_capture_enabled=True,
    )
    if field == "acquired_marks":
        state.acquired_marks[mark.locus] = mark
    payload = export_germline(state)
    payload[field].append(copy.deepcopy(payload[field][0]))

    with pytest.raises(
        ValueError,
        match=f"duplicate {field.removesuffix('_marks')} epigenetic mark locus",
    ):
        restore_germline(payload, genome)


def test_germline_restore_rejects_acquired_marks_when_capture_disabled():
    genome = _genome()
    state = GermlineState.from_genome(genome, acquired_capture_enabled=True)
    mark = EpigeneticMark("plasticity.learning_rate.baseline", delta=0.01)
    state.acquired_marks[mark.locus] = mark
    payload = export_germline(state)
    payload["acquired_capture_enabled"] = False

    with pytest.raises(ValueError, match="acquired marks require acquired capture"):
        restore_germline(payload, genome)


def test_germline_restore_rejects_birth_expression_not_derived_from_genome_and_marks():
    genome = _genome()
    state = GermlineState.from_genome(genome)
    payload = export_germline(state)
    locus = "plasticity.learning_rate.baseline"
    payload["birth_expression"][locus] += 0.001

    with pytest.raises(ValueError, match="birth expression does not match"):
        restore_germline(payload, genome)


def test_germline_restore_accepts_birth_expression_derived_with_inherited_mark():
    genome = _genome()
    mark = EpigeneticMark("plasticity.learning_rate.baseline", delta=0.01, strength=0.5)
    state = GermlineState.from_genome(genome, inherited_marks=(mark,))

    restored = restore_germline(export_germline(state), genome)

    assert restored.birth_expression == state.birth_expression


def test_expression_restore_rejects_birth_expression_not_derived_from_genome():
    genome = _genome()
    state = GeneExpressionState.from_genome(genome)
    payload = export_expression(state)
    locus = "plasticity.learning_rate.baseline"
    payload["birth_expression"][locus] += 0.001

    with pytest.raises(ValueError, match="birth expression does not match"):
        restore_expression(payload, genome)


def test_expression_restore_accepts_birth_expression_derived_from_inherited_germline():
    genome = _genome()
    mark = EpigeneticMark("plasticity.learning_rate.baseline", delta=0.01, strength=0.5)
    germline = GermlineState.from_genome(genome, inherited_marks=(mark,))
    expression = GeneExpressionState.from_genome(genome, germline=germline)
    restored = restore_expression(
        export_expression(expression),
        genome,
        germline=germline,
    )

    assert restored.as_dict() == expression.as_dict()


@pytest.mark.parametrize(
    "field,value",
    [("exploration_drive", 1.1), ("exploration_drive", float("nan"))],
)
def test_expression_state_restore_rejects_out_of_range_values(field, value):
    genome = _genome()
    payload = export_expression(GeneExpressionState.from_genome(genome))
    payload[field] = value
    with pytest.raises(ValueError):
        restore_expression_state(payload, genome)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_expression_state_restore_rejects_invalid_regulatory_activation(value):
    genome = _genome()
    payload = export_expression(GeneExpressionState.from_genome(genome))
    payload["regulatory_activation"]["mismatch_pressure"] = value
    with pytest.raises(ValueError):
        restore_expression_state(payload, genome)


def test_expression_restore_rejects_truncated_activation_after_update():
    genome = _genome()
    initial = GeneExpressionState.from_genome(genome)
    updated = ExpressionRegulator().update(
        genome, initial, RegulatorySignals(uncertainty=0.5)
    )
    payload = export_expression(updated)
    assert payload["update_count"] == 1
    assert set(payload["regulatory_activation"]) == {
        "mismatch_pressure", "resource_pressure"
    }
    payload["regulatory_activation"] = {}

    with pytest.raises(ValueError, match="fields do not match update_count"):
        restore_expression_state(payload, genome)


def test_expression_restore_rejects_activation_on_never_updated_state():
    genome = _genome()
    payload = export_expression(GeneExpressionState.from_genome(genome))
    payload["regulatory_activation"] = {"mismatch_pressure": 0.0}

    with pytest.raises(ValueError, match="fields do not match update_count"):
        restore_expression_state(payload, genome)
