from __future__ import annotations

import math
from dataclasses import dataclass, fields


def _require_finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return number


_LOWER_IS_BETTER = ("prediction_error", "representation_cost", "instability")

_HIGHER_IS_BETTER = ("information_retained", "calibration")


@dataclass(slots=True, frozen=True)
class LearningObjective:
    prediction_error: float
    representation_cost: float
    instability: float
    information_retained: float
    calibration: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite, got {value!r}")


def dominates(a: LearningObjective, b: LearningObjective) -> bool:
    at_least_as_good = True
    strictly_better_somewhere = False
    for field in _LOWER_IS_BETTER:
        a_value, b_value = getattr(a, field), getattr(b, field)
        if a_value > b_value:
            at_least_as_good = False
        elif a_value < b_value:
            strictly_better_somewhere = True
    for field in _HIGHER_IS_BETTER:
        a_value, b_value = getattr(a, field), getattr(b, field)
        if a_value < b_value:
            at_least_as_good = False
        elif a_value > b_value:
            strictly_better_somewhere = True
    return at_least_as_good and strictly_better_somewhere


@dataclass(slots=True)
class MetaParameter:
    value: float
    minimum: float
    maximum: float
    max_step: float

    def __post_init__(self) -> None:
        self.value = _require_finite(self.value, field="value")
        self.minimum = _require_finite(self.minimum, field="minimum")
        self.maximum = _require_finite(self.maximum, field="maximum")
        self.max_step = _require_finite(self.max_step, field="max_step", minimum=0.0)
        if self.minimum > self.maximum:
            raise ValueError(f"minimum ({self.minimum}) cannot exceed maximum ({self.maximum})")
        if not (self.minimum <= self.value <= self.maximum):
            raise ValueError(
                f"value ({self.value}) must be within [{self.minimum}, {self.maximum}]"
            )

    def propose(
        self,
        *,
        candidate_delta: float,
        previous_window_objective: LearningObjective,
        current_window_objective: LearningObjective,
        frozen: bool = False,
    ) -> None:
        if frozen:
            return  # safe mode (SafetyState.frozen) -- no parameter drift while frozen
        if dominates(previous_window_objective, current_window_objective):
            return  # things got worse -- never commit
        clipped_delta = max(-self.max_step, min(self.max_step, candidate_delta))
        self.value = max(self.minimum, min(self.maximum, self.value + clipped_delta))

    def checkpoint(self) -> dict[str, float]:
        return {
            "value": self.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "max_step": self.max_step,
        }

    @classmethod
    def restore(cls, payload: object) -> "MetaParameter":
        if not isinstance(payload, dict):
            raise ValueError("metaparameter checkpoint must be an object")
        expected_fields = {"value", "minimum", "maximum", "max_step"}
        if set(payload) != expected_fields:
            raise ValueError("metaparameter checkpoint fields do not match expected schema")
        return cls(
            value=_require_finite(payload["value"], field="value"),
            minimum=_require_finite(payload["minimum"], field="minimum"),
            maximum=_require_finite(payload["maximum"], field="maximum"),
            max_step=_require_finite(payload["max_step"], field="max_step", minimum=0.0),
        )



@dataclass(slots=True)
class SafetyState:
    consecutive_failures: int = 0
    frozen: bool = False

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.frozen = True

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.frozen = False
