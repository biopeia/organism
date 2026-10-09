"""Unit tests for OutcomeValueLedger — statistics, context, eviction, checkpoint."""

from __future__ import annotations

import math

import pytest

from organism.agency.value import (
    MAX_OUTCOME_VALUES,
    OutcomeValueLedger,
    context_key_for,
)


def test_empty_ledger_estimate_returns_none():
    ledger = OutcomeValueLedger()
    assert ledger.estimate("outcome.unknown") is None
    assert ledger.known_outcome_count == 0


def test_single_observation_produces_estimate():
    ledger = OutcomeValueLedger()
    ledger.observe("outcome.a", 0.5, tick=1)
    est = ledger.estimate("outcome.a")
    assert est is not None
    assert est.samples == 1
    assert math.isfinite(est.mean_value)
    assert est.variance == 0.0
    assert 0.0 < est.confidence <= 1.0


def test_welford_mean_converges():
    ledger = OutcomeValueLedger()
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for tick, v in enumerate(values):
        ledger.observe("outcome.b", v, tick=tick)
    est = ledger.estimate("outcome.b")
    assert est is not None
    assert abs(est.mean_value - 0.5) < 0.01
    assert est.samples == 5
    assert est.variance > 0.0


def test_welford_variance_increases_with_spread():
    ledger1 = OutcomeValueLedger()
    ledger2 = OutcomeValueLedger()
    for tick, v in enumerate([0.5, 0.5, 0.5]):
        ledger1.observe("outcome.tight", v, tick=tick)
    for tick, v in enumerate([-1.0, 0.0, 1.0]):
        ledger2.observe("outcome.spread", v, tick=tick)
    est1 = ledger1.estimate("outcome.tight")
    est2 = ledger2.estimate("outcome.spread")
    assert est1 is not None
    assert est2 is not None
    assert est1.variance < est2.variance


def test_confidence_saturates_at_16_samples():
    ledger = OutcomeValueLedger()
    for tick in range(16):
        ledger.observe("outcome.c", 0.1, tick=tick)
    est = ledger.estimate("outcome.c")
    assert est is not None
    assert est.confidence == 1.0


def test_confidence_grows_with_samples():
    ledger = OutcomeValueLedger()
    ledger.observe("outcome.d", 0.5, tick=0)
    for tick in range(1, 5):
        ledger.observe("outcome.d", 0.5, tick=tick)
    est4 = ledger.estimate("outcome.d")
    for tick in range(5, 16):
        ledger.observe("outcome.d", 0.5, tick=tick)
    est16 = ledger.estimate("outcome.d")
    assert est4 is not None
    assert est16 is not None
    assert est4.confidence <= est16.confidence


def test_clamping_to_minus_one_plus_one():
    ledger = OutcomeValueLedger()
    ledger.observe("outcome.extreme", 999.0, tick=0)
    ledger.observe("outcome.extreme", -999.0, tick=1)
    est = ledger.estimate("outcome.extreme")
    assert est is not None
    assert est.mean_value == 0.0


def test_nan_observation_silently_ignored():
    ledger = OutcomeValueLedger()
    ledger.observe("outcome.nan", float("nan"), tick=0)
    assert ledger.estimate("outcome.nan") is None
    assert ledger.known_outcome_count == 0


def test_inf_observation_silently_ignored():
    ledger = OutcomeValueLedger()
    ledger.observe("outcome.inf", float("inf"), tick=0)
    assert ledger.estimate("outcome.inf") is None


def test_capacity_enforced_at_256():
    ledger = OutcomeValueLedger()
    for i in range(MAX_OUTCOME_VALUES + 10):
        ledger.observe(f"outcome.{i:04d}", 0.5, tick=i)
    assert ledger.known_outcome_count <= MAX_OUTCOME_VALUES


def test_eviction_removes_lowest_evidence_entry():
    ledger = OutcomeValueLedger()
    for i in range(MAX_OUTCOME_VALUES):
        ledger.observe(f"outcome.{i:04d}", 0.5, tick=0)
        ledger.observe(f"outcome.{i:04d}", 0.5, tick=1)
    ledger.observe("outcome.new_weak", 0.5, tick=2)
    assert ledger.known_outcome_count == MAX_OUTCOME_VALUES


def test_same_outcome_can_have_opposite_value_in_two_private_contexts():
    ledger = OutcomeValueLedger()
    low = context_key_for(("sense.rec.103", "state.sense.x.level.1"))
    high = context_key_for(("sense.rec.103", "state.sense.x.level.6"))

    for tick in range(4):
        ledger.observe("outcome.same", 0.6, tick=tick, context_key=low)
        ledger.observe("outcome.same", -0.5, tick=tick, context_key=high)

    low_value = ledger.estimate("outcome.same", context_key=low)
    high_value = ledger.estimate("outcome.same", context_key=high)
    global_value = ledger.estimate("outcome.same")

    assert low_value is not None and low_value.mean_value > 0.0
    assert high_value is not None and high_value.mean_value < 0.0
    assert global_value is not None
    assert low_value.mean_value != high_value.mean_value


def test_unknown_context_falls_back_to_global_value():
    ledger = OutcomeValueLedger()
    known = context_key_for(("state.sense.a.level.1",))
    unknown = context_key_for(("state.sense.a.level.7",))
    ledger.observe("outcome.a", 0.4, tick=1, context_key=known)

    contextual_fallback = ledger.estimate("outcome.a", context_key=unknown)
    global_value = ledger.estimate("outcome.a")

    assert contextual_fallback is not None and global_value is not None
    assert contextual_fallback.mean_value == global_value.mean_value


def test_context_key_is_stable_and_order_sensitive():
    first = context_key_for(("a", "b"))
    assert first == context_key_for(("a", "b"))
    assert first != context_key_for(("b", "a"))
    assert first.startswith("value-context.")
    assert len(first.removeprefix("value-context.")) == 24


def test_checkpoint_roundtrip_preserves_statistics_and_context():
    ledger = OutcomeValueLedger()
    context = context_key_for(("state.sense.a.level.2",))
    for tick in range(8):
        ledger.observe(
            "outcome.roundtrip",
            tick * 0.1 - 0.4,
            tick=tick,
            context_key=context,
        )

    checkpoint = ledger.checkpoint()
    restored = OutcomeValueLedger.restore(checkpoint)

    original = ledger.estimate("outcome.roundtrip", context_key=context)
    roundtripped = restored.estimate("outcome.roundtrip", context_key=context)

    assert original is not None
    assert roundtripped is not None
    assert abs(original.mean_value - roundtripped.mean_value) < 1e-9
    assert abs(original.variance - roundtripped.variance) < 1e-9
    assert original.samples == roundtripped.samples
    assert original.confidence == roundtripped.confidence
    assert restored.contextual_outcome_count == 1


def test_restore_accepts_v1_global_checkpoint():
    checkpoint = {
        "schema_version": 1,
        "stats": {
            "outcome.valid": {
                "samples": 5,
                "mean": 0.3,
                "m2": 0.02,
                "positive": 3,
                "negative": 1,
                "last_tick": 10,
            }
        },
    }
    restored = OutcomeValueLedger.restore(checkpoint)
    estimate = restored.estimate("outcome.valid")
    assert estimate is not None and estimate.mean_value == pytest.approx(0.3)
    assert restored.contextual_outcome_count == 0


def test_restore_fails_closed_on_unknown_schema():
    with pytest.raises(ValueError, match="schema"):
        OutcomeValueLedger.restore({"schema_version": 99, "stats": {}})


def test_restore_fails_on_non_dict():
    with pytest.raises(ValueError):
        OutcomeValueLedger.restore("not a dict")


def test_restore_skips_corrupted_entries():
    checkpoint = {
        "schema_version": 1,
        "stats": {
            "outcome.valid": {
                "samples": 5,
                "mean": 0.3,
                "m2": 0.02,
                "positive": 3,
                "negative": 1,
                "last_tick": 10,
            },
            "outcome.corrupted": "not a dict",
        },
    }
    restored = OutcomeValueLedger.restore(checkpoint)
    assert restored.estimate("outcome.valid") is not None
    assert restored.estimate("outcome.corrupted") is None


def test_multiple_outcomes_independent():
    ledger = OutcomeValueLedger()
    ledger.observe("a", 0.9, tick=0)
    ledger.observe("b", -0.9, tick=0)
    est_a = ledger.estimate("a")
    est_b = ledger.estimate("b")
    assert est_a is not None
    assert est_b is not None
    assert est_a.mean_value > 0
    assert est_b.mean_value < 0
