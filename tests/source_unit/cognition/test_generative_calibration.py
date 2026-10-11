import pytest

from organism.cognition.generative import (
    GenerativeOperation,
    PredictionCalibration,
)


def test_calibration_requires_later_comparison_and_groups_bounded_bands():
    calibration = PredictionCalibration()
    calibration.record_prediction(
        model_id="m0", operation=GenerativeOperation.PREDICT, depth=5, uncertainty=0.2
    )
    calibration.record_comparison(
        model_id="m0",
        operation=GenerativeOperation.PREDICT,
        depth=5,
        uncertainty=0.2,
        observed_error=0.4,
    )
    bucket = calibration.bucket(
        model_id="m0", operation=GenerativeOperation.PREDICT, depth=5, uncertainty=0.2
    )
    assert bucket.prediction_count == 1
    assert bucket.later_comparable_count == 1
    assert bucket.calibration_error == 0.2


def test_calibration_checkpoint_round_trip_preserves_buckets():
    calibration = PredictionCalibration()
    calibration.record_prediction(
        model_id="m0", operation=GenerativeOperation.PREDICT, depth=5, uncertainty=0.2
    )
    calibration.record_comparison(
        model_id="m0",
        operation=GenerativeOperation.PREDICT,
        depth=5,
        uncertainty=0.2,
        observed_error=0.4,
    )

    restored = PredictionCalibration.from_checkpoint(calibration.checkpoint())

    assert restored.checkpoint() == calibration.checkpoint()


def test_calibration_checkpoint_rejects_duplicate_bucket_identity():
    calibration = PredictionCalibration()
    calibration.record_prediction(
        model_id="m0", operation=GenerativeOperation.PREDICT, depth=5, uncertainty=0.2
    )
    payload = calibration.checkpoint()
    duplicate = dict(payload["buckets"][0])
    duplicate["prediction_count"] = 2
    duplicate["uncertainty_sum"] = 0.4
    payload["buckets"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate calibration bucket"):
        PredictionCalibration.from_checkpoint(payload)


def test_calibration_remains_bounded_under_repeated_comparisons():
    import json
    calibration = PredictionCalibration()
    for _ in range(100):
        calibration.record_prediction(
            model_id="m0", operation=GenerativeOperation.PREDICT, depth=5, uncertainty=0.2
        )
        calibration.record_comparison(
            model_id="m0",
            operation=GenerativeOperation.PREDICT,
            depth=5,
            uncertainty=0.2,
            observed_error=0.3,
        )

    bucket = calibration.bucket(
        model_id="m0", operation=GenerativeOperation.PREDICT, depth=5, uncertainty=0.2
    )
    assert bucket.prediction_count == 100
    assert bucket.later_comparable_count == 100
    assert bucket.mean_observed_error == pytest.approx(0.3)
    assert bucket.calibration_error == pytest.approx(0.1)

    payload = calibration.checkpoint()
    # The bucket contains scalar sufficient statistics, not a 100-item list
    assert "error_sum" in payload["buckets"][0]
    assert "errors" not in payload["buckets"][0]
    serialized = json.dumps(payload)
    # 1 bucket with scalar fields is small (< 250 bytes) regardless of comparison count
    assert len(serialized) < 250


def test_calibration_from_legacy_checkpoint_with_errors_list():
    legacy_payload = {
        "buckets": [
            {
                "model_id": "m0",
                "operation": "predict",
                "depth_bucket": 1,
                "uncertainty_bucket": 2,
                "prediction_count": 2,
                "comparable_count": 2,
                "uncertainty_sum": 0.4,
                "errors": [0.2, 0.4],
            }
        ]
    }
    restored = PredictionCalibration.from_checkpoint(legacy_payload)
    bucket = restored.bucket(
        model_id="m0", operation=GenerativeOperation.PREDICT, depth=5, uncertainty=0.2
    )
    assert bucket.prediction_count == 2
    assert bucket.later_comparable_count == 2
    assert bucket.mean_observed_error == pytest.approx(0.3)
    # Checkpoint produces the new bounded sufficient statistics shape
    assert "error_sum" in restored.checkpoint()["buckets"][0]
