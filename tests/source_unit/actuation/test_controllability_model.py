"""Regression tests for the owner-ratified causal evidence floor."""

from organism.actuation.model import controllability_confidence


def test_zero_action_linked_hits_never_yield_positive_confidence():
    assert controllability_confidence(8, 0, 0, 0) == 0.0
    assert controllability_confidence(8, 0, 8, 0) == 0.0
    assert controllability_confidence(8, 0, 8, 8) == 0.0


def test_no_action_opportunities_yield_zero_confidence():
    assert controllability_confidence(0, 0, 8, 0) == 0.0
