"""Organism-owned composition for one bounded cognitive graph transition.

This composes opaque sensory activation and optional, explicitly configured
edge plasticity. It does not create graph structure, invent learning
parameters, publish actions, or claim to be a complete Organism transition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .graph import CognitiveGraph, GraphFrame, PlasticEdge, PlasticNode
from .learning import (
    PredictionError,
    apply_oja_update,
    compute_prediction_errors,
    update_eligibility,
)
from .sense_input import (
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    bind_opaque_sense_signals,
)
from .transition import CognitiveTransitionState, advance_cognition
from .types import EdgeKind, NodeKind


class CognitiveStage(StrEnum):
    ACTIVATION = "activation"
    PREDICTION_ERROR = "prediction_error"
    PLASTICITY = "plasticity"


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: CognitiveStage
    status: str


@dataclass(frozen=True, slots=True)
class EdgeLearningInstruction:
    """Explicit per-edge learning inputs; no rates or eligibility are inferred.

    The eligibility trace is updated from activations for the named edge.
    ``eligible`` separately gates whether its weight may change this step.
    """

    source_id: str
    target_id: str
    kind: EdgeKind
    learning_rate: float
    modulation: float
    eligibility_decay: float
    eligible: bool
    frozen: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_id, str)
            or not self.source_id
            or not isinstance(self.target_id, str)
            or not self.target_id
        ):
            raise ValueError("learning edge identity must be non-empty strings")
        if not isinstance(self.kind, EdgeKind):
            raise TypeError("learning edge kind must be an EdgeKind")
        for name in ("learning_rate", "modulation", "eligibility_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= self.eligibility_decay <= 1:
            raise ValueError("eligibility_decay must be within [0, 1]")
        if type(self.eligible) is not bool or type(self.frozen) is not bool:
            raise ValueError("eligible and frozen must be booleans")


@dataclass(frozen=True, slots=True)
class CognitionComposition:
    """Candidate result of the declared graph-composition scope."""

    state: CognitiveTransitionState
    graph: CognitiveGraph
    frame: GraphFrame
    prediction_errors: tuple[PredictionError, ...]
    stages: tuple[StageResult, ...]


def _copy_graph(graph: CognitiveGraph) -> CognitiveGraph:
    """Copy mutable edge state so a rejected candidate cannot mutate input."""
    return CognitiveGraph(
        nodes=tuple(
            PlasticNode(
                node_id=node.node_id,
                kind=node.kind,
                bias=node.bias,
                tau=node.tau,
                predicts_node_id=node.predicts_node_id,
            )
            for node in graph.nodes
        ),
        edges=tuple(
            PlasticEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                kind=edge.kind,
                weight=edge.weight,
                plasticity=edge.plasticity,
                delay_ticks=edge.delay_ticks,
                eligibility=edge.eligibility,
                support=edge.support,
                age_ticks=edge.age_ticks,
                stable_ticks=edge.stable_ticks,
                last_use_tick=edge.last_use_tick,
            )
            for edge in graph.edges
        ),
        kernel_limits=graph.kernel_limits,
    )


def compose_cognition_transition(
    state: CognitiveTransitionState,
    graph: CognitiveGraph,
    opaque_signals: Mapping[str, float],
    *,
    causal_tick: int,
    learning: tuple[EdgeLearningInstruction, ...] = (),
) -> CognitionComposition:
    """Compose activation, eligible prediction error, and optional learning.

    All work is staged against a copied graph and immutable transition state.
    The caller receives a candidate and decides whether to publish it as part
    of the wider Organism transition. Learning runs only for explicitly named
    edges and uses caller-supplied parameters; an empty policy means no update.
    """
    if not isinstance(learning, tuple) or any(
        not isinstance(item, EdgeLearningInstruction) for item in learning
    ):
        raise ValueError("learning must be a tuple of EdgeLearningInstruction values")
    keys = [(item.source_id, item.target_id, item.kind) for item in learning]
    if len(keys) != len(set(keys)):
        raise ValueError("learning instructions must not duplicate an edge")

    candidate_graph = _copy_graph(graph)
    next_state, frame = advance_cognition(
        state, candidate_graph, opaque_signals, causal_tick=causal_tick
    )
    previous = state.activation_mapping()
    has_predictors = any(
        node.kind is NodeKind.PREDICTOR for node in candidate_graph.nodes
    )
    errors = (
        compute_prediction_errors(
            candidate_graph, current=frame.activations, previous=previous
        )
        if has_predictors
        else ()
    )
    stages = [StageResult(CognitiveStage.ACTIVATION, "executed")]
    stages.append(
        StageResult(
            CognitiveStage.PREDICTION_ERROR,
            "executed"
            if errors
            else "waiting_for_history"
            if has_predictors
            else "not_applicable",
        )
    )

    if learning:
        for instruction in learning:
            edge = next(
                (
                    item
                    for item in candidate_graph.edges
                    if (item.source_id, item.target_id, item.kind)
                    == (instruction.source_id, instruction.target_id, instruction.kind)
                ),
                None,
            )
            if edge is None:
                raise ValueError("learning instruction references an undeclared edge")
            update_eligibility(
                edge,
                source_previous=previous.get(edge.source_id, 0.0),
                target_current=frame.activations.get(edge.target_id, 0.0),
                decay=float(instruction.eligibility_decay),
            )
            apply_oja_update(
                edge,
                source_activation=frame.activations.get(edge.source_id, 0.0),
                target_activation=frame.activations.get(edge.target_id, 0.0),
                learning_rate=float(instruction.learning_rate),
                modulation=float(instruction.modulation),
                eligible=instruction.eligible,
                frozen=instruction.frozen,
            )
        stages.append(StageResult(CognitiveStage.PLASTICITY, "executed"))
    else:
        stages.append(StageResult(CognitiveStage.PLASTICITY, "not_applicable"))

    return CognitionComposition(
        state=next_state,
        graph=candidate_graph,
        frame=frame,
        prediction_errors=errors,
        stages=tuple(stages),
    )


def compose_cognition_from_sense_signals(
    state: CognitiveTransitionState,
    graph: CognitiveGraph,
    bindings: tuple[SenseChannelBinding, ...],
    signals: tuple[OpaqueSenseSignal, ...],
    *,
    expected_provenance: SenseProvenance,
    learning: tuple[EdgeLearningInstruction, ...] = (),
) -> CognitionComposition:
    """Validate opaque inputs, then stage one cognition candidate.

    Input binding and cognition execution are kept within the same owner-side
    operation, so callers cannot accidentally publish a transition after
    validating a different signal set.
    """
    node_inputs = bind_opaque_sense_signals(
        graph, bindings, signals, expected_provenance=expected_provenance
    )
    return compose_cognition_transition(
        state,
        graph,
        node_inputs,
        causal_tick=expected_provenance.causal_tick,
        learning=learning,
    )
