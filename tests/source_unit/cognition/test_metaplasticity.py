from __future__ import annotations

import pytest

from organism.cognition.metaplasticity import (
    LearningObjective,
    MetaParameter,
    SafetyState,
    dominates,
)


def _objective(
    prediction_error=0.5,
    representation_cost=0.5,
    instability=0.5,
    information_retained=0.5,
    calibration=0.5,
) -> LearningObjective:
    return LearningObjective(
        prediction_error=prediction_error,
        representation_cost=representation_cost,
        instability=instability,
        information_retained=information_retained,
        calibration=calibration,
    )


def test_non_finite_objective_field_is_rejected():
    import math

    with pytest.raises(ValueError):
        _objective(prediction_error=math.nan)
    with pytest.raises(ValueError):
        _objective(information_retained=math.inf)


def test_identical_objectives_never_dominate_each_other():
    a = _objective()
    b = _objective()
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_strictly_better_on_all_dimensions_dominates():
    better = _objective(
        prediction_error=0.1,
        representation_cost=0.1,
        instability=0.1,
        information_retained=0.9,
        calibration=0.9,
    )
    worse = _objective(
        prediction_error=0.5,
        representation_cost=0.5,
        instability=0.5,
        information_retained=0.5,
        calibration=0.5,
    )
    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_mixed_improvement_and_regression_dominates_neither_way():
    mixed = _objective(prediction_error=0.1, representation_cost=0.9)
    baseline = _objective(prediction_error=0.5, representation_cost=0.5)
    assert not dominates(mixed, baseline)
    assert not dominates(baseline, mixed)


def test_better_on_one_dimension_equal_on_rest_dominates():
    better = _objective(prediction_error=0.1)
    baseline = _objective(prediction_error=0.5)
    assert dominates(better, baseline)


def test_meta_parameter_never_changes_while_frozen():
    param = MetaParameter(value=0.5, minimum=0.0, maximum=1.0, max_step=0.1)
    previous = _objective(prediction_error=0.5)
    current = _objective(prediction_error=0.1)
    param.propose(
        candidate_delta=0.05,
        previous_window_objective=previous,
        current_window_objective=current,
        frozen=True,
    )
    assert param.value == 0.5


def test_meta_parameter_commits_delta_when_current_dominates_previous():
    param = MetaParameter(value=0.5, minimum=0.0, maximum=1.0, max_step=0.1)
    previous = _objective(prediction_error=0.5)
    current = _objective(prediction_error=0.1)
    param.propose(
        candidate_delta=0.05, previous_window_objective=previous, current_window_objective=current
    )
    assert param.value == 0.55


def test_meta_parameter_commits_delta_when_neither_dominates_tie():
    param = MetaParameter(value=0.5, minimum=0.0, maximum=1.0, max_step=0.1)
    same = _objective()
    param.propose(
        candidate_delta=0.05, previous_window_objective=same, current_window_objective=same
    )
    assert param.value == 0.55


def test_meta_parameter_reverts_when_previous_dominates_current():
    param = MetaParameter(value=0.5, minimum=0.0, maximum=1.0, max_step=0.1)
    previous = _objective(prediction_error=0.1)
    current = _objective(prediction_error=0.5)
    param.propose(
        candidate_delta=0.05, previous_window_objective=previous, current_window_objective=current
    )
    assert param.value == 0.5


def test_meta_parameter_clips_delta_to_max_step():
    param = MetaParameter(value=0.5, minimum=0.0, maximum=1.0, max_step=0.1)
    same = _objective()
    param.propose(
        candidate_delta=10.0, previous_window_objective=same, current_window_objective=same
    )
    assert param.value == 0.6


def test_meta_parameter_never_leaves_its_range():
    param = MetaParameter(value=0.95, minimum=0.0, maximum=1.0, max_step=0.5)
    same = _objective()
    param.propose(
        candidate_delta=1.0, previous_window_objective=same, current_window_objective=same
    )
    assert param.value == 1.0


def test_safety_state_freezes_after_three_consecutive_failures():
    state = SafetyState()
    state.record_failure()
    state.record_failure()
    assert not state.frozen
    state.record_failure()
    assert state.frozen


def test_safety_state_success_resets_the_streak():
    state = SafetyState()
    state.record_failure()
    state.record_failure()
    state.record_success()
    state.record_failure()
    state.record_failure()
    assert not state.frozen


def test_safety_state_frozen_never_clears_without_explicit_reset():
    state = SafetyState()
    for _ in range(3):
        state.record_failure()
    assert state.frozen
    state.record_success()
    assert state.frozen
    state.reset()
    assert not state.frozen
    assert state.consecutive_failures == 0


def test_meta_parameter_checkpoint_and_restore_roundtrip():
    param = MetaParameter(value=0.75, minimum=0.1, maximum=0.9, max_step=0.05)
    payload = param.checkpoint()
    assert payload == {
        "value": 0.75,
        "minimum": 0.1,
        "maximum": 0.9,
        "max_step": 0.05,
    }

    restored = MetaParameter.restore(payload)
    assert restored.value == 0.75
    assert restored.minimum == 0.1
    assert restored.maximum == 0.9
    assert restored.max_step == 0.05


def test_meta_parameter_validation_and_restore_fail_closed():
    # Value out of range
    with pytest.raises(ValueError, match="value.*must be within"):
        MetaParameter(value=1.5, minimum=0.0, maximum=1.0, max_step=0.1)

    # Minimum > Maximum
    with pytest.raises(ValueError, match="minimum.*cannot exceed maximum"):
        MetaParameter(value=0.5, minimum=1.0, maximum=0.0, max_step=0.1)

    # Negative max_step
    with pytest.raises(ValueError, match="max_step must be at least 0.0"):
        MetaParameter(value=0.5, minimum=0.0, maximum=1.0, max_step=-0.1)

    # Non-finite values
    with pytest.raises(ValueError, match="value must be finite"):
        MetaParameter(value=float("nan"), minimum=0.0, maximum=1.0, max_step=0.1)

    # Restore bad payload type
    with pytest.raises(ValueError, match="checkpoint must be an object"):
        MetaParameter.restore([0.5, 0.0, 1.0, 0.1])

    # Restore missing/extra fields
    with pytest.raises(ValueError, match="fields do not match expected schema"):
        MetaParameter.restore({"value": 0.5, "minimum": 0.0, "maximum": 1.0})

    with pytest.raises(ValueError, match="fields do not match expected schema"):
        MetaParameter.restore(
            {"value": 0.5, "minimum": 0.0, "maximum": 1.0, "max_step": 0.1, "extra": 1}
        )

    # Restore boolean / invalid types
    with pytest.raises(ValueError, match="value must be a number"):
        MetaParameter.restore({"value": True, "minimum": 0.0, "maximum": 1.0, "max_step": 0.1})
