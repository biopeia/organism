from __future__ import annotations

import copy

import pytest

from organism.genetics import (
    EpigeneticMark,
    GermlineState,
    GeneExpressionState,
    create_standard_genome,
    export_expression,
    export_germline,
    restore_expression,
    restore_germline,
)


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
