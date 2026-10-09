"""Organism-owned, explicit-time cognitive transition state.

This boundary advances the extracted cognitive graph over opaque modality
signals. Readouts remain consultative: this module does not choose or execute
body actions. The returned state is an in-memory value; it is not by itself a
portable checkpoint or a complete Organism continuation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .graph import CognitiveGraph, GraphFrame, TickContext
from .types import NodeKind
from ..identity import OrganismIdentity


class CognitiveTransitionError(ValueError):
    """Raised when an explicit cognitive transition is invalid."""


@dataclass(frozen=True, slots=True)
class CognitiveTransitionState:
    """Organism-private recurrent activation state for the graph kernel.

    Values are represented as a sorted tuple so callers cannot mutate the
    previous frame through an alias. Its component checkpoint is deliberately
    narrower than a complete Organism cognition checkpoint.
    """

    identity: OrganismIdentity
    last_causal_tick: int | None = None
    previous_activations: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.last_causal_tick is not None and (
            isinstance(self.last_causal_tick, bool)
            or not isinstance(self.last_causal_tick, int)
            or self.last_causal_tick < 0
        ):
            raise CognitiveTransitionError("last_causal_tick must be a non-negative integer")

        names: set[str] = set()
        for item in self.previous_activations:
            if not isinstance(item, tuple) or len(item) != 2:
                raise CognitiveTransitionError("previous_activations must contain name/value pairs")
            node_id, activation = item
            if not isinstance(node_id, str) or not node_id:
                raise CognitiveTransitionError("activation node ids must be non-empty strings")
            if node_id in names:
                raise CognitiveTransitionError(f"duplicate activation node id {node_id!r}")
            if isinstance(activation, bool) or not isinstance(activation, (int, float)):
                raise CognitiveTransitionError(f"activation for {node_id!r} must be numeric")
            if not math.isfinite(activation):
                raise CognitiveTransitionError(f"activation for {node_id!r} must be finite")
            names.add(node_id)
        if self.previous_activations != tuple(sorted(self.previous_activations)):
            raise CognitiveTransitionError("previous_activations must be sorted by node id")

    @classmethod
    def initial(cls, identity: OrganismIdentity) -> CognitiveTransitionState:
        """Create empty in-memory state for an explicitly identified Organism."""

        return cls(identity=identity)

    def activation_mapping(self) -> dict[str, float]:
        """Return a fresh mapping for the graph kernel; no mutable alias escapes."""

        return dict(self.previous_activations)

    def checkpoint_payload(self) -> dict[str, Any]:
        """Export this recurrent transition state as owner-private data.

        This is one cognition component only; it does not include graph
        structure/weights, learned models, or other Organism-owned state.
        """
        return {
            "schema": "biopeia.organism.cognitive-transition",
            "schema_version": 1,
            "identity": str(self.identity),
            "last_causal_tick": self.last_causal_tick,
            "previous_activations": [list(item) for item in self.previous_activations],
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: object) -> CognitiveTransitionState:
        """Restore transition state from a strictly shaped owner payload."""
        if not isinstance(payload, dict) or set(payload) != {
            "schema", "schema_version", "identity", "last_causal_tick", "previous_activations"
        }:
            raise CognitiveTransitionError("invalid cognitive transition checkpoint shape")
        if payload["schema"] != "biopeia.organism.cognitive-transition" or payload[
            "schema_version"
        ] != 1:
            raise CognitiveTransitionError("unsupported cognitive transition checkpoint")
        raw_activations = payload["previous_activations"]
        if not isinstance(raw_activations, list):
            raise CognitiveTransitionError("previous_activations must be a list")
        try:
            identity = OrganismIdentity.parse(payload["identity"])
            activations = tuple(
                (item[0], item[1])
                for item in raw_activations
                if isinstance(item, list) and len(item) == 2
            )
            if len(activations) != len(raw_activations):
                raise CognitiveTransitionError("invalid activation checkpoint entry")
            return cls(identity, payload["last_causal_tick"], activations)
        except CognitiveTransitionError:
            raise
        except (TypeError, ValueError) as exc:
            raise CognitiveTransitionError("invalid cognitive transition checkpoint") from exc

    def checkpoint_bytes(self) -> bytes:
        """Encode this cognition component deterministically for an envelope."""
        return json.dumps(
            self.checkpoint_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def from_checkpoint_bytes(cls, payload: bytes) -> CognitiveTransitionState:
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CognitiveTransitionError("invalid cognitive transition checkpoint JSON") from exc
        return cls.from_checkpoint_payload(decoded)


def advance_cognition(
    state: CognitiveTransitionState,
    graph: CognitiveGraph,
    opaque_signals: Mapping[str, float],
    *,
    causal_tick: int,
) -> tuple[CognitiveTransitionState, GraphFrame]:
    """Advance one Organism-owned cognitive transition at an explicit World tick.

    Only declared SENSE nodes may receive inputs. A failed validation or graph
    activation returns no new state, leaving the caller's prior immutable value
    unchanged. This function neither owns World commit nor publishes actions.
    """

    if isinstance(causal_tick, bool) or not isinstance(causal_tick, int) or causal_tick < 0:
        raise CognitiveTransitionError("causal_tick must be a non-negative integer")
    if state.last_causal_tick is not None and causal_tick <= state.last_causal_tick:
        raise CognitiveTransitionError(
            "causal_tick must be greater than the last committed Organism cognitive tick"
        )

    signals: dict[str, float] = {}
    for node_id, value in opaque_signals.items():
        node = graph.node_by_id(node_id)
        if node is None or node.kind is not NodeKind.SENSE:
            raise CognitiveTransitionError(
                f"opaque signal {node_id!r} must target a declared SENSE node"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CognitiveTransitionError(f"opaque signal {node_id!r} must be numeric")
        if not math.isfinite(value):
            raise CognitiveTransitionError(f"opaque signal {node_id!r} must be finite")
        signals[node_id] = float(value)

    frame = graph.activate(
        signals,
        TickContext(causal_tick),
        previous=state.activation_mapping(),
    )
    next_state = CognitiveTransitionState(
        identity=state.identity,
        last_causal_tick=causal_tick,
        previous_activations=tuple(sorted(frame.activations.items())),
    )
    return next_state, frame

