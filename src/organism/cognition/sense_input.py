"""Explicit binding and correlation for opaque external SENSE inputs.

This boundary does not interpret a signal. It binds an owner-issued channel
identity to a declared SENSE node and rejects samples that do not match the
expected World/source/body applicability for the current transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .graph import CognitiveGraph, GraphFrame
from .transition import (
    CognitiveTransitionError,
    CognitiveTransitionState,
    advance_cognition,
)
from .types import NodeKind


SCALAR_FIELD_CONTRACT_ID = "biopeia.modality.dimensionless-scalar-field-identity"
SCALAR_FIELD_CONTRACT_REVISION = 1
SCALAR_FIELD_QUANTITY_KIND = "dimensionless_scalar_field"
SCALAR_FIELD_UNIT = "1"


@dataclass(frozen=True, slots=True)
class SenseProvenance:
    """Typed applicability references supplied by the causal owner boundary.

    These identifiers are correlation data, not claims that Organism owns or
    independently verifies the referenced World, Environment, or Body state.
    """

    world_id: str
    causal_tick: int
    physical_step_id: str
    environment_id: str
    environment_revision: str
    sample_frame_id: str
    sample_position_m: tuple[float, float, float]
    body_id: str
    embodiment_episode_id: str
    morphology_revision: str

    def __post_init__(self) -> None:
        for name in (
            "world_id",
            "physical_step_id",
            "environment_id",
            "environment_revision",
            "sample_frame_id",
            "body_id",
            "embodiment_episode_id",
            "morphology_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CognitiveTransitionError(f"{name} must be a non-empty string")
        if not isinstance(self.sample_position_m, tuple) or len(self.sample_position_m) != 3:
            raise CognitiveTransitionError("sample_position_m must be a 3D position tuple")
        for coordinate in self.sample_position_m:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise CognitiveTransitionError(
                    "sample_position_m coordinates must be finite numeric meters"
                )
            try:
                finite = math.isfinite(coordinate)
            except OverflowError:
                finite = False
            if not finite:
                raise CognitiveTransitionError(
                    "sample_position_m coordinates must be finite numeric meters"
                )
        if (
            isinstance(self.causal_tick, bool)
            or not isinstance(self.causal_tick, int)
            or self.causal_tick < 0
        ):
            raise CognitiveTransitionError("causal_tick must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OpaqueSenseSignal:
    """One numeric signal with explicit transduction and applicability identity."""

    channel_id: str
    receptor_id: str
    value: int | float
    provenance: SenseProvenance
    contract_id: str
    contract_revision: int
    quantity_kind: str
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.channel_id, str) or not self.channel_id:
            raise CognitiveTransitionError("channel_id must be a non-empty string")
        if not isinstance(self.receptor_id, str) or not self.receptor_id:
            raise CognitiveTransitionError("receptor_id must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise CognitiveTransitionError("opaque sense value must be numeric, not boolean")
        try:
            finite = math.isfinite(self.value)
        except OverflowError:
            finite = False
        if not finite or not 0.0 <= self.value <= 1.0:
            raise CognitiveTransitionError("opaque scalar-field value must be finite and in [0, 1]")
        if self.contract_id != SCALAR_FIELD_CONTRACT_ID:
            raise CognitiveTransitionError("unsupported scalar-field contract")
        if (
            isinstance(self.contract_revision, bool)
            or not isinstance(self.contract_revision, int)
            or self.contract_revision != SCALAR_FIELD_CONTRACT_REVISION
        ):
            raise CognitiveTransitionError("unsupported scalar-field contract revision")
        if self.quantity_kind != SCALAR_FIELD_QUANTITY_KIND or self.unit != SCALAR_FIELD_UNIT:
            raise CognitiveTransitionError("incompatible scalar-field quantity or unit")
        if not isinstance(self.provenance, SenseProvenance):
            raise CognitiveTransitionError("provenance must be a SenseProvenance")


@dataclass(frozen=True, slots=True)
class SenseChannelBinding:
    """Explicit channel/receptor-to-private-SENSE-node binding.

    Channel and receptor identities remain distinct: the former identifies
    the transduced signal, while the latter identifies the Embodiment-owned
    physical receiver.
    """

    channel_id: str
    receptor_id: str
    sense_node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.channel_id, str) or not self.channel_id:
            raise CognitiveTransitionError("binding channel_id must be a non-empty string")
        if not isinstance(self.receptor_id, str) or not self.receptor_id:
            raise CognitiveTransitionError("binding receptor_id must be a non-empty string")
        if not isinstance(self.sense_node_id, str) or not self.sense_node_id:
            raise CognitiveTransitionError("sense_node_id must be a non-empty string")


def advance_cognition_from_sense_signals(
    state: CognitiveTransitionState,
    graph: CognitiveGraph,
    bindings: tuple[SenseChannelBinding, ...],
    signals: tuple[OpaqueSenseSignal, ...],
    *,
    expected_provenance: SenseProvenance,
) -> tuple[CognitiveTransitionState, GraphFrame]:
    """Validate and bind a complete configured signal set before advancing.

    A missing configured channel is an error, never an implicit numeric zero.
    The expected correlation is supplied by the World/owner composition for
    the source cut; this check detects stale or mismatched material, but is not
    authentication and does not make Organism authoritative over those owners.
    """

    node_inputs = bind_opaque_sense_signals(
        graph, bindings, signals, expected_provenance=expected_provenance
    )
    return advance_cognition(
        state, graph, node_inputs, causal_tick=expected_provenance.causal_tick
    )


def bind_opaque_sense_signals(
    graph: CognitiveGraph,
    bindings: tuple[SenseChannelBinding, ...],
    signals: tuple[OpaqueSenseSignal, ...],
    *,
    expected_provenance: SenseProvenance,
) -> dict[str, float]:
    """Validate a configured signal set and bind it to private SENSE nodes.

    This returns only opaque numeric inputs. It deliberately does not advance
    cognition, allowing the owner composition to stage validation, cognition,
    and canonical state publication as one transition.
    """
    if not isinstance(graph, CognitiveGraph):
        raise CognitiveTransitionError("graph must be a CognitiveGraph")
    if not isinstance(expected_provenance, SenseProvenance):
        raise CognitiveTransitionError("expected_provenance must be a SenseProvenance")
    if not isinstance(bindings, tuple) or not isinstance(signals, tuple):
        raise CognitiveTransitionError("bindings and signals must be explicit tuples")
    binding_by_channel: dict[str, SenseChannelBinding] = {}
    bound_nodes: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, SenseChannelBinding):
            raise CognitiveTransitionError("bindings must contain SenseChannelBinding values")
        if binding.channel_id in binding_by_channel:
            raise CognitiveTransitionError(f"duplicate channel binding {binding.channel_id!r}")
        if binding.sense_node_id in bound_nodes:
            raise CognitiveTransitionError(f"duplicate SENSE node binding {binding.sense_node_id!r}")
        node = graph.node_by_id(binding.sense_node_id)
        if node is None or node.kind is not NodeKind.SENSE:
            raise CognitiveTransitionError(
                f"binding target {binding.sense_node_id!r} must be a declared SENSE node"
            )
        binding_by_channel[binding.channel_id] = binding
        bound_nodes.add(binding.sense_node_id)

    if not binding_by_channel:
        raise CognitiveTransitionError("at least one explicit SENSE channel binding is required")
    graph_sense_nodes = {
        node.node_id for node in graph.nodes if node.kind is NodeKind.SENSE
    }
    if bound_nodes != graph_sense_nodes:
        unbound = ", ".join(sorted(graph_sense_nodes - bound_nodes))
        extra = ", ".join(sorted(bound_nodes - graph_sense_nodes))
        raise CognitiveTransitionError(
            "bindings must cover exactly the graph's declared SENSE nodes"
            + (f"; unbound: {unbound}" if unbound else "")
            + (f"; unexpected: {extra}" if extra else "")
        )
    signal_by_channel: dict[str, OpaqueSenseSignal] = {}
    for signal in signals:
        if not isinstance(signal, OpaqueSenseSignal):
            raise CognitiveTransitionError("signals must contain OpaqueSenseSignal values")
        if signal.channel_id in signal_by_channel:
            raise CognitiveTransitionError(f"duplicate signal channel {signal.channel_id!r}")
        if signal.channel_id not in binding_by_channel:
            raise CognitiveTransitionError(f"unconfigured signal channel {signal.channel_id!r}")
        if signal.receptor_id != binding_by_channel[signal.channel_id].receptor_id:
            raise CognitiveTransitionError(
                f"signal channel {signal.channel_id!r} has mismatched receptor identity"
            )
        if signal.provenance != expected_provenance:
            raise CognitiveTransitionError(
                f"signal channel {signal.channel_id!r} has stale or mismatched provenance"
            )
        signal_by_channel[signal.channel_id] = signal

    missing = set(binding_by_channel) - set(signal_by_channel)
    if missing:
        names = ", ".join(sorted(missing))
        raise CognitiveTransitionError(f"missing configured SENSE signal(s): {names}")

    node_inputs = {
        binding_by_channel[channel_id].sense_node_id: float(signal.value)
        for channel_id, signal in signal_by_channel.items()
    }
    return node_inputs


__all__ = [
    "OpaqueSenseSignal",
    "SCALAR_FIELD_CONTRACT_ID",
    "SCALAR_FIELD_CONTRACT_REVISION",
    "SCALAR_FIELD_QUANTITY_KIND",
    "SCALAR_FIELD_UNIT",
    "SenseChannelBinding",
    "SenseProvenance",
    "advance_cognition_from_sense_signals",
    "bind_opaque_sense_signals",
]
