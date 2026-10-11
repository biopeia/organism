"""Bounded calibration statistics for later-comparable predictions."""

from __future__ import annotations

from dataclasses import dataclass

from .types import GenerativeOperation, bounded_identifier, unit_interval


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    prediction_count: int
    later_comparable_count: int
    mean_declared_uncertainty: float
    mean_observed_error: float
    calibration_error: float


class PredictionCalibration:
    """Stores aggregate calibration only after an external comparison."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, GenerativeOperation, int, int], list[float]] = {}

    def record_prediction(
        self,
        *,
        model_id: str,
        operation: GenerativeOperation,
        depth: int,
        uncertainty: float,
    ) -> None:
        self._validate(model_id, operation, depth, uncertainty)
        key = self._key(model_id, operation, depth, uncertainty)
        bucket = self._buckets.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        bucket[0] += 1
        bucket[2] += uncertainty

    def record_comparison(
        self,
        *,
        model_id: str,
        operation: GenerativeOperation,
        depth: int,
        uncertainty: float,
        observed_error: float,
    ) -> None:
        self._validate(model_id, operation, depth, uncertainty)
        unit_interval(observed_error, name="observed_error")
        key = self._key(model_id, operation, depth, uncertainty)
        bucket = self._buckets.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        if bucket[1] >= bucket[0]:
            raise ValueError("comparison requires an unpaired prediction in the same bucket")
        bucket[1] += 1
        bucket[3] += observed_error

    def checkpoint(self) -> dict[str, object]:
        """Persist bounded aggregate calibration sufficient statistics, never unbounded lists."""
        return {
            "buckets": [
                {
                    "model_id": model_id,
                    "operation": operation.value,
                    "depth_bucket": depth_bucket,
                    "uncertainty_bucket": uncertainty_bucket,
                    "prediction_count": int(values[0]),
                    "comparable_count": int(values[1]),
                    "uncertainty_sum": float(values[2]),
                    "error_sum": float(values[3]),
                }
                for (model_id, operation, depth_bucket, uncertainty_bucket), values in sorted(
                    self._buckets.items(),
                    key=lambda item: (
                        item[0][0],
                        item[0][1].value,
                        item[0][2],
                        item[0][3],
                    ),
                )
            ]
        }

    @classmethod
    def from_checkpoint(cls, payload: object) -> "PredictionCalibration":
        if not isinstance(payload, dict):
            raise ValueError("calibration checkpoint must be an object")
        if set(payload) != {"buckets"}:
            raise ValueError("calibration checkpoint fields are incomplete or unknown")
        raw_buckets = payload["buckets"]
        if not isinstance(raw_buckets, list) or len(raw_buckets) > 4096:
            raise ValueError("invalid calibration bucket collection")
        obj = cls()
        for item in raw_buckets:
            if not isinstance(item, dict):
                raise ValueError("invalid calibration bucket")
            # Support both new bounded shape (error_sum) and historical shape (errors)
            has_error_sum = "error_sum" in item
            expected_fields_bounded = {
                "model_id", "operation", "depth_bucket", "uncertainty_bucket",
                "prediction_count", "comparable_count", "uncertainty_sum", "error_sum",
            }
            expected_fields_legacy = {
                "model_id", "operation", "depth_bucket", "uncertainty_bucket",
                "prediction_count", "comparable_count", "uncertainty_sum", "errors",
            }
            if set(item) != expected_fields_bounded and set(item) != expected_fields_legacy:
                raise ValueError("invalid calibration bucket shape")
            try:
                model_id = bounded_identifier(item["model_id"], name="model_id")
                operation = GenerativeOperation(item["operation"])
                depth_bucket = _required_int(item["depth_bucket"], "depth_bucket")
                uncertainty_bucket = _required_int(item["uncertainty_bucket"], "uncertainty_bucket")
                prediction_count = _required_int(item["prediction_count"], "prediction_count")
                comparable_count = _required_int(item["comparable_count"], "comparable_count")
                uncertainty_sum = _required_number(item["uncertainty_sum"], "uncertainty_sum")
                if has_error_sum:
                    error_sum = _required_number(item["error_sum"], "error_sum")
                else:
                    if not isinstance(item["errors"], list):
                        raise ValueError("errors must be a list")
                    legacy_errors = [_required_number(value, "error") for value in item["errors"]]
                    if len(legacy_errors) != comparable_count or any(not 0.0 <= error <= 1.0 for error in legacy_errors):
                        raise ValueError("invalid legacy calibration bucket errors")
                    error_sum = sum(legacy_errors)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid calibration bucket") from exc
            if (
                depth_bucket < 0
                or not 0 <= uncertainty_bucket <= 9
                or prediction_count < 0
                or comparable_count < 0
                or comparable_count > prediction_count
                or not 0.0 <= uncertainty_sum <= float(prediction_count)
                or not 0.0 <= error_sum <= float(comparable_count)
            ):
                raise ValueError("invalid calibration bucket values")
            key = (model_id, operation, depth_bucket, uncertainty_bucket)
            if key in obj._buckets:
                raise ValueError("duplicate calibration bucket")
            obj._buckets[key] = [
                float(prediction_count),
                float(comparable_count),
                uncertainty_sum,
                float(error_sum),
            ]
        return obj

    def bucket(
        self, *, model_id: str, operation: GenerativeOperation, depth: int, uncertainty: float
    ) -> CalibrationBucket:
        self._validate(model_id, operation, depth, uncertainty)
        values = self._buckets.get(
            self._key(model_id, operation, depth, uncertainty), [0.0, 0.0, 0.0, 0.0]
        )
        prediction_count, comparable_count, uncertainty_sum, error_sum = values[:4]
        mean_uncertainty = uncertainty_sum / prediction_count if prediction_count else 0.0
        mean_error = error_sum / comparable_count if comparable_count else 0.0
        return CalibrationBucket(
            int(prediction_count),
            int(comparable_count),
            mean_uncertainty,
            mean_error,
            abs(mean_uncertainty - mean_error) if comparable_count else 0.0,
        )

    @staticmethod
    def _key(model_id, operation, depth, uncertainty):
        return (model_id, operation, depth // 4, min(9, int(uncertainty * 10)))

    @staticmethod
    def _validate(model_id, operation, depth, uncertainty):
        bounded_identifier(model_id, name="model_id")
        if not isinstance(operation, GenerativeOperation):
            raise ValueError("operation must be a GenerativeOperation")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        unit_interval(uncertainty, name="uncertainty")


def _required_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


__all__ = ["CalibrationBucket", "PredictionCalibration"]
