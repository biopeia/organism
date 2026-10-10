from __future__ import annotations

from uuid import UUID

import pytest

from organism.cognition.composition import (
    CognitiveStage,
    EdgeLearningInstruction,
    compose_cognition_transition,
)
from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.transition import CognitiveTransitionState
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity


def _graph(*, predictor: bool = False) -> CognitiveGraph:
    nodes = [
        PlasticNode("opaque.channel.1", NodeKind.SENSE),
        PlasticNode("private.state.1", NodeKind.STATE),
    ]
    edges = [
        PlasticEdge(
            source_id="opaque.channel.1",
            target_id="private.state.1",
            kind=EdgeKind.EXCITATORY,
            weight=0.5,
            plasticity=0.5,
            delay_ticks=1,
        )
    ]
    if predictor:
        nodes.append(
            PlasticNode(
                "private.predictor.1",
                NodeKind.PREDICTOR,
                predicts_node_id="private.state.1",
            )
        )
        edges.append(
            PlasticEdge(
                source_id="private.state.1",
                target_id="private.predictor.1",
                kind=EdgeKind.PREDICTIVE,
                weight=0.5,
                plasticity=0.0,
                delay_ticks=1,
            )
        )
    return CognitiveGraph(
        nodes=tuple(nodes), edges=tuple(edges), kernel_limits=KernelLimits()
    )


def _state() -> CognitiveTransitionState:
    return CognitiveTransitionState.initial(
        OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000123"))
    )


def test_composition_executes_opaque_sense_graph_and_reports_inapplicable_stages() -> (
    None
):
    graph = _graph()
    state = _state()

    result = compose_cognition_transition(
        state, graph, {"opaque.channel.1": 0.75}, causal_tick=3
    )

    assert result.frame.tick == 3
    assert result.frame.activations["opaque.channel.1"] == pytest.approx(0.75)
    assert result.state.last_causal_tick == 3
    assert [(stage.stage, stage.status) for stage in result.stages] == [
        (CognitiveStage.ACTIVATION, "executed"),
        (CognitiveStage.PREDICTION_ERROR, "not_applicable"),
        (CognitiveStage.PLASTICITY, "not_applicable"),
    ]
    assert state.last_causal_tick is None
    assert graph.edges[0].weight == pytest.approx(0.5)


def test_composition_computes_predictor_errors_after_activation() -> None:
    graph = _graph(predictor=True)
    state = _state()
    first = compose_cognition_transition(
        state, graph, {"opaque.channel.1": 0.8}, causal_tick=4
    )
    assert (first.stages[1].stage, first.stages[1].status) == (
        CognitiveStage.PREDICTION_ERROR,
        "waiting_for_history",
    )
    second = compose_cognition_transition(
        first.state, first.graph, {"opaque.channel.1": 0.2}, causal_tick=5
    )

    assert second.prediction_errors
    assert second.prediction_errors[0].target_id == "private.state.1"
    assert (second.stages[1].stage, second.stages[1].status) == (
        CognitiveStage.PREDICTION_ERROR,
        "executed",
    )


def test_explicit_plasticity_updates_only_candidate_graph() -> None:
    graph = _graph()
    initial = compose_cognition_transition(
        _state(), graph, {"opaque.channel.1": 0.8}, causal_tick=0
    )
    result = compose_cognition_transition(
        initial.state,
        initial.graph,
        {"opaque.channel.1": 0.6},
        causal_tick=1,
        learning=(
            EdgeLearningInstruction(
                "opaque.channel.1",
                "private.state.1",
                kind=EdgeKind.EXCITATORY,
                learning_rate=0.1,
                modulation=1.0,
                eligibility_decay=0.5,
                eligible=True,
            ),
        ),
    )

    assert (result.stages[-1].stage, result.stages[-1].status) == (
        CognitiveStage.PLASTICITY,
        "executed",
    )
    assert result.graph.edges[0].weight != pytest.approx(graph.edges[0].weight)
    assert graph.edges[0].weight == pytest.approx(0.5)


def test_learning_selects_parallel_edge_by_kind() -> None:
    graph = CognitiveGraph(
        nodes=(
            PlasticNode("opaque.channel.1", NodeKind.SENSE),
            PlasticNode("private.state.1", NodeKind.STATE),
        ),
        edges=(
            PlasticEdge(
                "opaque.channel.1",
                "private.state.1",
                EdgeKind.EXCITATORY,
                weight=0.1,
                plasticity=0.5,
                delay_ticks=0,
            ),
            PlasticEdge(
                "opaque.channel.1",
                "private.state.1",
                EdgeKind.INHIBITORY,
                weight=-0.5,
                plasticity=0.5,
                delay_ticks=0,
            ),
        ),
        kernel_limits=KernelLimits(),
    )

    result = compose_cognition_transition(
        _state(),
        graph,
        {"opaque.channel.1": 0.8},
        causal_tick=0,
        learning=(
            EdgeLearningInstruction(
                "opaque.channel.1",
                "private.state.1",
                kind=EdgeKind.INHIBITORY,
                learning_rate=0.1,
                modulation=1.0,
                eligibility_decay=0.5,
                eligible=True,
            ),
        ),
    )

    assert result.graph.edges[0].weight == pytest.approx(0.1)
    assert result.graph.edges[1].weight != pytest.approx(-0.5)


def test_learning_without_action_linked_eligibility_does_not_change_weight() -> None:
    graph = _graph()
    result = compose_cognition_transition(
        _state(),
        graph,
        {"opaque.channel.1": 0.8},
        causal_tick=0,
        learning=(
            EdgeLearningInstruction(
                "opaque.channel.1",
                "private.state.1",
                kind=EdgeKind.EXCITATORY,
                learning_rate=0.1,
                modulation=1.0,
                eligibility_decay=0.5,
                eligible=False,
            ),
        ),
    )

    assert result.graph.edges[0].weight == pytest.approx(graph.edges[0].weight)
    assert graph.edges[0].weight == pytest.approx(0.5)


def test_invalid_signal_or_learning_instruction_does_not_mutate_prior_graph() -> None:
    graph = _graph()
    state = _state()
    with pytest.raises(ValueError):
        compose_cognition_transition(
            state,
            graph,
            {"opaque.channel.1": float("nan")},
            causal_tick=0,
        )
    with pytest.raises(ValueError, match="undeclared edge"):
        compose_cognition_transition(
            state,
            graph,
            {"opaque.channel.1": 0.5},
            causal_tick=0,
            learning=(
                EdgeLearningInstruction(
                    "missing",
                    "edge",
                    kind=EdgeKind.EXCITATORY,
                    learning_rate=0.1,
                    modulation=1.0,
                    eligibility_decay=0.5,
                    eligible=True,
                ),
            ),
        )
    assert state.last_causal_tick is None
    assert graph.edges[0].weight == pytest.approx(0.5)
    assert graph.edges[0].eligibility == pytest.approx(0.0)
