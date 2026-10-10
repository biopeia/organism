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
