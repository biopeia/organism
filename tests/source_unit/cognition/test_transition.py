from __future__ import annotations

import math
from uuid import UUID

import pytest

from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.transition import (
    CognitiveTransitionError,
    CognitiveTransitionState,
    advance_cognition,
)
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity


def _graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=(
            PlasticNode("sense", NodeKind.SENSE),
            PlasticNode("memory", NodeKind.STATE),
        ),
        edges=(
            PlasticEdge(
                source_id="sense",
                target_id="memory",
                kind=EdgeKind.EXCITATORY,
                weight=1.0,
                plasticity=0.0,
                delay_ticks=1,
            ),
        ),
        kernel_limits=KernelLimits(),
    )


def _state() -> CognitiveTransitionState:
    return CognitiveTransitionState.initial(
        OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000123"))
    )


def test_transition_uses_explicit_tick_and_prior_activation() -> None:
    graph = _graph()
    initial = _state()

    after_one, first = advance_cognition(initial, graph, {"sense": 0.8}, causal_tick=4)
    after_two, second = advance_cognition(after_one, graph, {"sense": 0.0}, causal_tick=5)

    assert first.tick == 4
    assert second.tick == 5
    assert after_two.identity == initial.identity
    assert after_two.last_causal_tick == 5
    assert second.activations["memory"] == pytest.approx(math.tanh(first.activations["sense"]))
    assert initial.last_causal_tick is None
    assert initial.previous_activations == ()


@pytest.mark.parametrize("bad_tick", [-1, True, 1.5])
def test_transition_rejects_invalid_tick_without_changing_prior_state(bad_tick: object) -> None:
    state = _state()
    with pytest.raises(CognitiveTransitionError):
        advance_cognition(state, _graph(), {"sense": 0.5}, causal_tick=bad_tick)  # type: ignore[arg-type]
    assert state.last_causal_tick is None
    assert state.previous_activations == ()


def test_transition_rejects_non_sense_or_unknown_signal_nodes() -> None:
    state = _state()
    for node_id in ("memory", "unknown"):
        with pytest.raises(CognitiveTransitionError, match="declared SENSE"):
            advance_cognition(state, _graph(), {node_id: 0.1}, causal_tick=0)


@pytest.mark.parametrize("bad_signal", [float("nan"), float("inf"), True, "1"])
def test_transition_rejects_nonfinite_or_nonnumeric_signal(bad_signal: object) -> None:
    with pytest.raises(CognitiveTransitionError):
        advance_cognition(_state(), _graph(), {"sense": bad_signal}, causal_tick=0)  # type: ignore[dict-item]


def test_transition_rejects_repeated_or_decreasing_causal_tick() -> None:
    state, _ = advance_cognition(_state(), _graph(), {"sense": 0.1}, causal_tick=10)
    for tick in (9, 10):
        with pytest.raises(CognitiveTransitionError, match="greater than"):
            advance_cognition(state, _graph(), {"sense": 0.2}, causal_tick=tick)

