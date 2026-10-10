import pytest

from organism.statistics import PairAccumulator


def test_pair_accumulator_checkpoint_round_trip():
    accumulator = PairAccumulator()
    accumulator.observe(1.0, 2.0)
    accumulator.observe(2.0, 4.0)
    accumulator.observe(3.0, 6.0)

    restored = PairAccumulator.from_payload(accumulator.to_payload())

    assert restored == accumulator
    assert restored.correlation == pytest.approx(1.0)


def test_pair_accumulator_checkpoint_rejects_unknown_field():
    accumulator = PairAccumulator()

    with pytest.raises(ValueError, match="fields do not match schema"):
        PairAccumulator.from_payload({**accumulator.to_payload(), "unknown": 1})


def test_pair_accumulator_checkpoint_rejects_missing_field():
    payload = PairAccumulator().to_payload()
    del payload["c_xy"]

    with pytest.raises(ValueError, match="fields do not match schema"):
        PairAccumulator.from_payload(payload)


def test_pair_accumulator_checkpoint_rejects_non_mapping_payload():
    with pytest.raises(ValueError, match="fields do not match schema"):
        PairAccumulator.from_payload([])  # type: ignore[arg-type]
