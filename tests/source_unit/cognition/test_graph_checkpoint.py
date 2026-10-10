from __future__ import annotations

import pytest

from organism.cognition.checkpoint import (
    export_graph_checkpoint,
    restore_graph_checkpoint,
)
from organism.cognition.graph import (
    CognitiveGraph,
    GraphError,
    PlasticEdge,
    PlasticNode,
)
from organism.cognition.limits import KernelLimits
from organism.cognition.types import EdgeKind, NodeKind


def _sense(node_id: str = "s") -> PlasticNode:
    return PlasticNode(node_id=node_id, kind=NodeKind.SENSE)


def _concept(node_id: str = "c") -> PlasticNode:
    return PlasticNode(node_id=node_id, kind=NodeKind.CONCEPT, bias=0.2, tau=1.5)


def _edge(weight: float = 0.75, eligibility: float = 3.2) -> PlasticEdge:
    edge = PlasticEdge(
        source_id="s",
        target_id="c",
        kind=EdgeKind.EXCITATORY,
        weight=weight,
        plasticity=0.5,
        delay_ticks=0,
    )
    edge.eligibility = eligibility
    edge.support = 42
    edge.age_ticks = 17
    edge.last_use_tick = 9
    return edge


def test_none_graph_round_trips_to_none():
    assert export_graph_checkpoint(None) is None
    assert restore_graph_checkpoint(None, kernel_limits=KernelLimits()) is None


def test_exported_payload_round_trips_topology_and_static_fields():
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    payload = export_graph_checkpoint(graph)
    restored = restore_graph_checkpoint(payload, kernel_limits=KernelLimits())

    assert {n.node_id for n in restored.nodes} == {"s", "c"}
    restored_concept = next(n for n in restored.nodes if n.node_id == "c")
    assert restored_concept.bias == pytest.approx(0.2)
    assert restored_concept.tau == pytest.approx(1.5)

    restored_edge = restored.edges[0]
    assert restored_edge.source_id == "s"
    assert restored_edge.target_id == "c"
    assert restored_edge.plasticity == pytest.approx(0.5)
    assert restored_edge.support == 42
    assert restored_edge.age_ticks == 17
    assert restored_edge.last_use_tick == 9


@pytest.mark.parametrize("weight", [0.4, 0.009, 0.05, -1.7, -0.0])
@pytest.mark.parametrize("eligibility", [0.4, 0.009, 0.05, -1.7, -10.0, 10.0])
def test_weight_and_eligibility_round_trip_exactly(weight, eligibility):
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()),
        edges=(_edge(weight=weight, eligibility=eligibility),),
        kernel_limits=KernelLimits(),
    )
    payload = export_graph_checkpoint(graph)
    restored = restore_graph_checkpoint(payload, kernel_limits=KernelLimits())
    restored_edge = restored.edges[0]
    assert payload["weight_codec_version"] == 4
    assert restored_edge.weight == weight
    assert restored_edge.eligibility == eligibility


def test_consecutive_checkpoints_preserve_exact_live_weight_changes():
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(weight=0.750001),), kernel_limits=KernelLimits()
    )
    before = export_graph_checkpoint(graph)
    graph.edges[0].weight = 0.750002  # a change far smaller than one quantization bin
    after = export_graph_checkpoint(graph)
    assert before["edges"][0]["weight"] == 0.750001
    assert after["edges"][0]["weight"] == 0.750002


def test_integer_edge_values_export_as_restorable_floats():
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(weight=1, eligibility=0),),
        kernel_limits=KernelLimits(),
    )
    payload = export_graph_checkpoint(graph)
    assert type(payload["edges"][0]["weight"]) is float
    assert type(payload["edges"][0]["eligibility"]) is float
    restored = restore_graph_checkpoint(payload, kernel_limits=KernelLimits())
    assert restored.edges[0].weight == 1.0
    assert restored.edges[0].eligibility == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("weight", float("nan")), ("weight", 2.01), ("eligibility", float("nan")),
     ("eligibility", 10.01)],
)
def test_export_rejects_mutated_invalid_edge_state(field, value):
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    setattr(graph.edges[0], field, value)
    with pytest.raises(GraphError, match=field):
        export_graph_checkpoint(graph)


def test_exact_graph_restore_preserves_identical_future_learning_transitions():
    from organism.cognition.learning import apply_oja_update, update_eligibility

    graph = CognitiveGraph(
        nodes=(_sense(), _concept()),
        edges=(_edge(weight=0.400001, eligibility=0.05),),
        kernel_limits=KernelLimits(),
    )
    restored = restore_graph_checkpoint(
        export_graph_checkpoint(graph), kernel_limits=KernelLimits()
    )

    for source, target in ((0.2, 0.4), (0.7, 0.3), (-0.1, 0.8)):
        for candidate in (graph, restored):
            edge = candidate.edges[0]
            update_eligibility(edge, source_previous=source, target_current=target, decay=0.83)
            apply_oja_update(
                edge,
                source_activation=source,
                target_activation=target,
                learning_rate=0.07,
                modulation=0.6,
                eligible=True,
            )
        assert restored.edges[0].weight == graph.edges[0].weight
        assert restored.edges[0].eligibility == graph.edges[0].eligibility


def test_restore_rejects_a_payload_that_would_violate_kernel_limits():
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    payload = export_graph_checkpoint(graph)
    with pytest.raises(GraphError):
        restore_graph_checkpoint(payload, kernel_limits=KernelLimits(max_nodes=1))


def test_restore_rejects_malformed_payload_type():
    with pytest.raises(GraphError):
        restore_graph_checkpoint({"nodes": "not-a-list", "edges": []}, kernel_limits=KernelLimits())


@pytest.mark.parametrize("codec_version", [None, 1, 2, 3, 99])
def test_restore_rejects_missing_or_noncurrent_weight_codec_version(codec_version):
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    payload = export_graph_checkpoint(graph)
    if codec_version is None:
        del payload["weight_codec_version"]
    else:
        payload["weight_codec_version"] = codec_version
    with pytest.raises(GraphError, match="codec version"):
        restore_graph_checkpoint(payload, kernel_limits=KernelLimits())


def test_weight_codec_rejects_non_finite_live_weights():
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    payload = export_graph_checkpoint(graph)
    payload["edges"][0]["weight"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        restore_graph_checkpoint(payload, kernel_limits=KernelLimits())


@pytest.mark.parametrize("field", ["weight", "eligibility"])
@pytest.mark.parametrize("value", [True, 1, float("nan"), float("inf")])
def test_restore_rejects_invalid_exact_edge_state(field, value):
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    payload = export_graph_checkpoint(graph)
    payload["edges"][0][field] = value
    with pytest.raises(GraphError):
        restore_graph_checkpoint(payload, kernel_limits=KernelLimits())


@pytest.mark.parametrize("scope", ["graph", "node", "edge"])
def test_restore_rejects_unknown_checkpoint_fields(scope):
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    payload = export_graph_checkpoint(graph)
    target = payload if scope == "graph" else payload["nodes"][0] if scope == "node" else payload["edges"][0]
    target["unexpected"] = "value"
    with pytest.raises(GraphError, match="unknown or missing"):
        restore_graph_checkpoint(payload, kernel_limits=KernelLimits())


def test_restore_rejects_eligibility_outside_learning_bound():
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()), edges=(_edge(),), kernel_limits=KernelLimits()
    )
    payload = export_graph_checkpoint(graph)
    payload["edges"][0]["eligibility"] = 10.01
    with pytest.raises(GraphError, match="eligibility"):
        restore_graph_checkpoint(payload, kernel_limits=KernelLimits())


# --- SafetyState checkpoint ---

from organism.cognition.checkpoint import export_safety_state, restore_safety_state
from organism.cognition.metaplasticity import SafetyState


def test_safety_state_none_round_trips_to_a_fresh_state():
    restored = restore_safety_state(None)
    assert restored.consecutive_failures == 0
    assert not restored.frozen


def test_frozen_safety_state_survives_round_trip():
    state = SafetyState()
    for _ in range(3):
        state.record_failure()
    assert state.frozen

    payload = export_safety_state(state)
    restored = restore_safety_state(payload)

    assert restored.frozen  # a restart must not silently un-freeze a legitimately frozen network
    assert restored.consecutive_failures == 3


# --- SensoryNormalizer checkpoint ---

from organism.cognition.activation import SensoryNormalizer
from organism.cognition.checkpoint import (
    export_sensory_normalizers,
    restore_sensory_normalizers,
)


def test_unestablished_normalizer_is_not_exported():
    normalizers = {"s": SensoryNormalizer()}
    normalizers["s"].normalize(1.0)  # only 1 sample, not established
    payload = export_sensory_normalizers(normalizers)
    assert "s" not in payload


def test_established_normalizer_round_trips():
    normalizer = SensoryNormalizer()
    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        normalizer.normalize(value)
    payload = export_sensory_normalizers({"s": normalizer})
    restored = restore_sensory_normalizers(payload)
    assert restored["s"].mean == pytest.approx(normalizer.mean)
    assert restored["s"].variance == pytest.approx(normalizer.variance)
    assert restored["s"].count == normalizer.count


def test_tentative_weight_is_exact_in_codec_v4():
    graph = CognitiveGraph(
        nodes=(_sense(), _concept()),
        edges=(_edge(weight=0.05, eligibility=0.0),),
        kernel_limits=KernelLimits(),
    )

    payload = export_graph_checkpoint(graph)
    restored = restore_graph_checkpoint(payload, kernel_limits=KernelLimits())

    assert payload["weight_codec_version"] == 4
    assert payload["edges"][0]["weight"] == 0.05
    assert restored.edges[0].weight != 0.0
    assert restored.edges[0].weight == 0.05
