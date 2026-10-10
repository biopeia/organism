from __future__ import annotations

import math
from typing import Any

from organism.genetics.genome import (
    Genome,
    GenomeCodec,
    GenomeError,
    _genome_to_plain_dict,
)

from .activation import SensoryNormalizer
from .graph import CognitiveGraph, GraphError, PlasticEdge, PlasticNode
from .limits import KernelLimits
from .metaplasticity import SafetyState
from .types import WEIGHT_RANGE

WEIGHT_CODEC_VERSION = 4


def export_genome_checkpoint(genome: Genome | None) -> dict[str, Any] | None:
    if genome is None:
        return None
    payload = _genome_to_plain_dict(genome)
    payload["genome_hash"] = genome.genome_hash
    payload["genotype_hash"] = genome.genotype_hash
    return payload


def restore_genome_checkpoint(
    payload: dict[str, Any] | None,
    *,
    kernel_limits: KernelLimits,
    running_version: tuple[int, int, int],
) -> Genome | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or "genome_hash" not in payload:
        raise GenomeError("genome checkpoint payload must be an object with a genome_hash")
    persisted_hash = payload["genome_hash"]
    persisted_genotype_hash = payload.get("genotype_hash")
    if not isinstance(persisted_genotype_hash, str):
        raise GenomeError("genome checkpoint hashes are required")
    genome_fields = {
        key: value for key, value in payload.items() if key not in {"genome_hash", "genotype_hash"}
    }
    codec = GenomeCodec()
    genome = codec.load(genome_fields)
    if genome.genome_hash != persisted_hash:
        raise GenomeError("genome checkpoint hash mismatch -- payload may be corrupted or tampered")
    if persisted_genotype_hash != genome.genotype_hash:
        raise GenomeError(
            "genotype checkpoint hash mismatch -- payload may be corrupted or tampered"
        )
    codec.validate(genome, kernel_limits, running_version=running_version)
    return genome


def _require_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphError(f"{field} must be a number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise GraphError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise GraphError(f"{field} must be finite")
    return number


def _require_finite_float(value: Any, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise GraphError(f"{field} must be a finite float")
    return value


def _require_nonneg_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphError(f"{field} must be a non-negative integer")
    if value < 0:
        raise GraphError(f"{field} must be non-negative")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphError(f"{field} must be an integer")
    return value


def export_graph_checkpoint(
    graph: CognitiveGraph | None,
) -> dict[str, Any] | None:
    """Serialize the exact graph-owned state needed by this codec."""
    if graph is None:
        return None
    from .learning import ELIGIBILITY_BOUND

    edges: list[dict[str, Any]] = []
    for edge in graph.edges:
        weight = _require_finite(edge.weight, "edge.weight")
        if not WEIGHT_RANGE[0] <= weight <= WEIGHT_RANGE[1]:
            raise GraphError(
                f"edge.weight must be within [{WEIGHT_RANGE[0]}, {WEIGHT_RANGE[1]}]"
            )
        eligibility = _require_finite(edge.eligibility, "edge.eligibility")
        if abs(eligibility) > ELIGIBILITY_BOUND:
            raise GraphError(
                f"edge.eligibility must be within [-{ELIGIBILITY_BOUND}, {ELIGIBILITY_BOUND}]"
            )
        edges.append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "kind": edge.kind.value,
                "weight": weight,
                "plasticity": edge.plasticity,
                "delay_ticks": edge.delay_ticks,
                "eligibility": eligibility,
                "support": edge.support,
                "age_ticks": edge.age_ticks,
                "stable_ticks": edge.stable_ticks,
                "last_use_tick": edge.last_use_tick,
            }
        )
    return {
        "weight_codec_version": WEIGHT_CODEC_VERSION,
        "nodes": [
            {
                "node_id": node.node_id,
                "kind": node.kind.value,
                "bias": node.bias,
                "tau": node.tau,
                "predicts_node_id": node.predicts_node_id,
            }
            for node in graph.nodes
        ],
        "edges": edges,
    }


def restore_graph_checkpoint(
    payload: dict[str, Any] | None, *, kernel_limits: KernelLimits
) -> CognitiveGraph | None:
    """Re-validate untrusted checkpoint state fully before construction."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise GraphError("graph checkpoint payload must be an object")
    codec_version = payload.get("weight_codec_version")
    if (
        isinstance(codec_version, bool)
        or not isinstance(codec_version, int)
        or codec_version != WEIGHT_CODEC_VERSION
    ):
        raise GraphError(f"graph checkpoint requires weight codec version {WEIGHT_CODEC_VERSION}")

    from .types import EdgeKind, NodeKind

    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise GraphError("graph checkpoint payload must have list-shaped 'nodes' and 'edges'")

    if set(payload) != {"weight_codec_version", "nodes", "edges"}:
        raise GraphError("graph checkpoint payload has unknown or missing fields")
    node_fields = {"node_id", "kind", "bias", "tau", "predicts_node_id"}
    edge_fields = {
        "source_id", "target_id", "kind", "weight", "plasticity", "delay_ticks",
        "eligibility", "support", "age_ticks", "stable_ticks", "last_use_tick",
    }
    if any(not isinstance(entry, dict) or set(entry) != node_fields for entry in raw_nodes):
        raise GraphError("graph checkpoint nodes have unknown or missing fields")
    if any(not isinstance(entry, dict) or set(entry) != edge_fields for entry in raw_edges):
        raise GraphError("graph checkpoint edges have unknown or missing fields")

    from .learning import ELIGIBILITY_BOUND

    try:
        nodes = tuple(
            PlasticNode(
                node_id=str(entry["node_id"]),
                kind=NodeKind(entry["kind"]),
                bias=_require_finite(entry["bias"], "node.bias"),
                tau=_require_finite(entry["tau"], "node.tau"),
                predicts_node_id=entry.get("predicts_node_id"),
            )
            for entry in raw_nodes
        )
        edges = tuple(
            PlasticEdge(
                source_id=str(entry["source_id"]),
                target_id=str(entry["target_id"]),
                kind=EdgeKind(entry["kind"]),
                weight=_require_finite_float(entry["weight"], "edge.weight"),
                plasticity=_require_finite(entry["plasticity"], "edge.plasticity"),
                delay_ticks=_require_int(entry["delay_ticks"], "edge.delay_ticks"),
                eligibility=_require_finite_float(entry["eligibility"], "edge.eligibility"),
                support=_require_nonneg_int(entry["support"], "edge.support"),
                age_ticks=_require_nonneg_int(entry["age_ticks"], "edge.age_ticks"),
                stable_ticks=_require_nonneg_int(entry["stable_ticks"], "edge.stable_ticks"),
                last_use_tick=_require_nonneg_int(entry["last_use_tick"], "edge.last_use_tick"),
            )
            for entry in raw_edges
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, GraphError):
            raise
        raise GraphError(f"malformed graph checkpoint: {exc}") from exc
    if any(abs(edge.eligibility) > ELIGIBILITY_BOUND for edge in edges):
        raise GraphError(f"edge.eligibility must be within [-{ELIGIBILITY_BOUND}, {ELIGIBILITY_BOUND}]")
    return CognitiveGraph(nodes=nodes, edges=edges, kernel_limits=kernel_limits)


def export_safety_state(state: SafetyState) -> dict[str, Any]:
    return {"consecutive_failures": state.consecutive_failures, "frozen": state.frozen}


def restore_safety_state(payload: dict[str, Any] | None) -> SafetyState:
    if payload is None:
        return SafetyState()
    if not isinstance(payload, dict):
        raise GraphError("safety_state checkpoint must be an object")
    if set(payload) != {"consecutive_failures", "frozen"}:
        raise GraphError("safety_state checkpoint fields do not match the current schema")
    try:
        failures = _require_nonneg_int(
            payload["consecutive_failures"], "safety_state.consecutive_failures"
        )
        frozen = payload["frozen"]
    except KeyError as exc:
        raise GraphError(f"malformed safety_state checkpoint: missing {exc.args[0]!r}") from exc
    if not isinstance(frozen, bool):
        raise GraphError("safety_state.frozen must be a boolean")
    return SafetyState(consecutive_failures=failures, frozen=frozen)


def export_sensory_normalizers(normalizers: dict[str, SensoryNormalizer]) -> dict[str, Any]:
    """Export complete normalizer state for exact runtime continuation."""
    payload = {
        sense_id: {
            "mean": normalizer.mean,
            "variance": normalizer.variance,
            "count": normalizer.count,
        }
        for sense_id, normalizer in normalizers.items()
    }
    restore_sensory_normalizers(payload)
    return payload


def restore_sensory_normalizers(payload: dict[str, Any]) -> dict[str, SensoryNormalizer]:
    if not isinstance(payload, dict):
        raise GraphError("sensory_normalizers checkpoint must be an object")
    restored: dict[str, SensoryNormalizer] = {}
    for sense_id, entry in payload.items():
        if not isinstance(sense_id, str) or not sense_id:
            raise GraphError("sensory normalizer identity must be a non-empty string")
        if not isinstance(entry, dict):
            raise GraphError(f"sensory normalizer {sense_id!r} must be an object")
        if set(entry) != {"mean", "variance", "count"}:
            raise GraphError(
                f"sensory normalizer {sense_id!r} must contain exactly mean, variance, and count"
            )
        count = _require_nonneg_int(entry.get("count"), f"{sense_id}.count")
        variance = _require_finite(entry["variance"], f"{sense_id}.variance")
        if variance < 0:
            raise GraphError(f"{sense_id}.variance must be non-negative")
        restored[sense_id] = SensoryNormalizer(
            mean=_require_finite(entry["mean"], f"{sense_id}.mean"),
            variance=variance,
            count=count,
        )
    return restored
