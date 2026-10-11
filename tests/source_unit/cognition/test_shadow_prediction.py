import pytest
from organism.cognition.learning import (
    ComposedShadowPrediction,
    LaggedShadowPrediction,
    ShadowPrediction,
)


def test_shadow_prediction_requires_out_of_sample_gain():
    candidate = ShadowPrediction("a", "b")
    for _ in range(8):
        candidate.observe(1.0, 1.0, 0.0)
    assert candidate.samples == 8
    assert candidate.predictive_gain > 0
    assert candidate.promotable


def test_shadow_prediction_does_not_promote_without_gain():
    candidate = ShadowPrediction("a", "b")
    for _ in range(8):
        candidate.observe(0.0, 1.0, 0.0)
    assert not candidate.promotable


def test_shadow_prediction_checkpoint_and_restore_roundtrip():
    candidate = ShadowPrediction("src_node", "tgt_node")
    for _ in range(10):
        candidate.observe(1.0, 1.0, 0.0)

    payload = candidate.checkpoint()
    restored = ShadowPrediction.restore(payload)

    assert restored.source_id == "src_node"
    assert restored.target_id == "tgt_node"
    assert restored.samples == 10
    assert restored.status == candidate.status
    assert restored.predictive_gain == candidate.predictive_gain
    assert restored.checkpoint() == payload


def test_shadow_prediction_validation_and_restore_fail_closed():
    # Construction validation
    with pytest.raises(ValueError, match="source_id must be a non-empty string"):
        ShadowPrediction("", "tgt")

    with pytest.raises(ValueError, match="samples must be an integer"):
        ShadowPrediction("src", "tgt", samples=True)

    with pytest.raises(ValueError, match="samples must be non-negative"):
        ShadowPrediction("src", "tgt", samples=-1)

    with pytest.raises(ValueError, match="model_loss must be finite"):
        ShadowPrediction("src", "tgt", model_loss=float("nan"))

    with pytest.raises(ValueError, match="model_loss must be non-negative"):
        ShadowPrediction("src", "tgt", model_loss=-0.5)

    with pytest.raises(ValueError, match="status must be one of"):
        ShadowPrediction("src", "tgt", status="unknown_status")

    # Restore validation
    with pytest.raises(ValueError, match="checkpoint payload must be a Mapping"):
        ShadowPrediction.restore([1, 2, 3])

    with pytest.raises(ValueError, match="checkpoint fields do not match expected schema"):
        ShadowPrediction.restore({"source_id": "s", "target_id": "t"})

    with pytest.raises(ValueError, match="checkpoint fields do not match expected schema"):
        ShadowPrediction.restore(
            {
                "source_id": "s",
                "target_id": "t",
                "samples": 0,
                "model_loss": 0.0,
                "persistence_loss": 0.0,
                "status": "candidate",
                "extra": 123,
            }
        )


def test_lagged_shadow_prediction_checkpoint_and_restore_roundtrip():
    candidate = LaggedShadowPrediction("x", "y", lag_ticks=2)
    candidate.observe(1.0, 0.5)
    candidate.observe(2.0, 1.0)
    candidate.observe(3.0, 1.5)
    candidate.observe(4.0, 2.0)

    payload = candidate.checkpoint()
    restored = LaggedShadowPrediction.restore(payload)

    assert restored.source_id == "x"
    assert restored.target_id == "y"
    assert restored.lag_ticks == 2
    assert restored.samples == candidate.samples
    assert restored.model_loss == candidate.model_loss
    assert restored.persistence_loss == candidate.persistence_loss
    assert restored.status == candidate.status
    assert restored.checkpoint() == payload

    # Continue observation on both and ensure state stays identical
    candidate.observe(5.0, 2.5)
    restored.observe(5.0, 2.5)
    assert restored.checkpoint() == candidate.checkpoint()


def test_lagged_shadow_prediction_validation_and_restore_fail_closed():
    with pytest.raises(ValueError, match="lag_ticks must be an integer"):
        LaggedShadowPrediction("x", "y", lag_ticks=True)

    with pytest.raises(ValueError, match="lag_ticks must be within"):
        LaggedShadowPrediction("x", "y", lag_ticks=0)

    with pytest.raises(ValueError, match="lag_ticks must be within"):
        LaggedShadowPrediction("x", "y", lag_ticks=17)

    # Restore fail closed
    payload = LaggedShadowPrediction("x", "y", lag_ticks=2).checkpoint()

    bad_payload = dict(payload)
    bad_payload["source_history"] = [1.0, 2.0, 3.0, 4.0, 5.0]  # exceeds lag_ticks + 1
    with pytest.raises(ValueError, match="source_history exceeds maximum capacity"):
        LaggedShadowPrediction.restore(bad_payload)

    bad_payload2 = dict(payload)
    bad_payload2["target_history"] = [1.0, 2.0, 3.0]  # exceeds 2
    with pytest.raises(ValueError, match="target_history exceeds maximum capacity"):
        LaggedShadowPrediction.restore(bad_payload2)

    bad_payload3 = dict(payload)
    bad_payload3["status"] = "invalid"
    with pytest.raises(ValueError, match="status must be one of"):
        LaggedShadowPrediction.restore(bad_payload3)


def test_composed_shadow_prediction_checkpoint_and_restore_roundtrip():
    candidate = ComposedShadowPrediction("x", "m", "y", window_ticks=16)
    for i in range(12):
        candidate.observe(float(i), float(i * 2), float(i * 4))

    payload = candidate.checkpoint()
    restored = ComposedShadowPrediction.restore(payload)

    assert restored.source_id == "x"
    assert restored.intermediate_id == "m"
    assert restored.target_id == "y"
    assert restored.window_ticks == 16
    assert restored.samples == candidate.samples
    assert restored.model_loss == candidate.model_loss
    assert restored.persistence_loss == candidate.persistence_loss
    assert restored.status == candidate.status
    assert restored.first_relation_slope == candidate.first_relation_slope
    assert restored.second_relation_slope == candidate.second_relation_slope
    assert restored.checkpoint() == payload

    # Continue observation on both and ensure state stays identical
    candidate.observe(13.0, 26.0, 52.0)
    restored.observe(13.0, 26.0, 52.0)
    assert restored.checkpoint() == candidate.checkpoint()


def test_composed_shadow_prediction_validation_and_restore_fail_closed():
    with pytest.raises(ValueError, match="window_ticks must be within"):
        ComposedShadowPrediction("x", "m", "y", window_ticks=4)

    payload = ComposedShadowPrediction("x", "m", "y", window_ticks=16).checkpoint()

    bad_payload = dict(payload)
    bad_payload["first_pairs"] = [[1.0]]  # not 2-element
    with pytest.raises(ValueError, match="first_pairs items must be 2-element"):
        ComposedShadowPrediction.restore(bad_payload)

    bad_payload2 = dict(payload)
    bad_payload2["first_pairs"] = [[1.0, float("nan")]]
    with pytest.raises(ValueError, match="first_pairs item\\[1\\] must be finite"):
        ComposedShadowPrediction.restore(bad_payload2)

    bad_payload3 = dict(payload)
    bad_payload3["first_pairs"] = [[1.0, 2.0]] * 17  # exceeds window_ticks
    with pytest.raises(ValueError, match="first_pairs exceeds window_ticks capacity"):
        ComposedShadowPrediction.restore(bad_payload3)
