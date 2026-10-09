"""Bounded portable composition for the extracted cognitive graph runtime.

This composes graph structure/quantized learned weights with recurrent
activation state. It is not a complete cognition or Organism checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from organism.checkpoint import CheckpointDependency, OrganismCheckpoint
from organism.identity import OrganismIdentity

from .checkpoint import export_graph_checkpoint, restore_graph_checkpoint
from .graph import CognitiveGraph, GraphError
from .limits import KernelLimits
from .transition import CognitiveTransitionError, CognitiveTransitionState


class CognitiveRuntimeCheckpointError(ValueError):
    """Raised when a composed cognitive runtime checkpoint is invalid."""


@dataclass(frozen=True, slots=True)
class CognitiveRuntimeCheckpoint:
    """A graph plus its recurrent Organism-owned transition state."""

    state: CognitiveTransitionState
    graph: CognitiveGraph

    def __post_init__(self) -> None:
        if not isinstance(self.state, CognitiveTransitionState):
            raise CognitiveRuntimeCheckpointError("state must be CognitiveTransitionState")
        if not isinstance(self.graph, CognitiveGraph):
            raise CognitiveRuntimeCheckpointError("graph must be CognitiveGraph")

    def to_bytes(self) -> bytes:
        payload = {
            "schema": "biopeia.organism.cognitive-runtime",
            "schema_version": 1,
            "identity": str(self.state.identity),
            "transition": self.state.checkpoint_payload(),
            "graph": export_graph_checkpoint(self.graph),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(
        cls, payload: bytes, *, kernel_limits: KernelLimits
    ) -> CognitiveRuntimeCheckpoint:
        try:
            document = json.loads(payload)
            if not isinstance(document, dict) or set(document) != {
                "schema", "schema_version", "identity", "transition", "graph"
            }:
                raise CognitiveRuntimeCheckpointError("invalid cognitive runtime checkpoint shape")
            if document["schema"] != "biopeia.organism.cognitive-runtime" or document[
                "schema_version"
            ] != 1:
                raise CognitiveRuntimeCheckpointError("unsupported cognitive runtime checkpoint")
            identity = OrganismIdentity.parse(document["identity"])
            state = CognitiveTransitionState.from_checkpoint_payload(document["transition"])
            if state.identity != identity:
                raise CognitiveRuntimeCheckpointError("identity mismatch in cognitive checkpoint")
            graph = restore_graph_checkpoint(document["graph"], kernel_limits=kernel_limits)
            if graph is None:
                raise CognitiveRuntimeCheckpointError("graph is required")
            return cls(state=state, graph=graph)
        except CognitiveRuntimeCheckpointError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, GraphError,
                CognitiveTransitionError) as exc:
            raise CognitiveRuntimeCheckpointError("malformed cognitive runtime checkpoint") from exc

    def to_organism_checkpoint(
        self, *, dependencies: tuple[CheckpointDependency, ...] = ()
    ) -> OrganismCheckpoint:
        """Wrap this owner payload in the generic Organism checkpoint envelope."""
        return OrganismCheckpoint(self.state.identity, self.to_bytes(), dependencies)

    @classmethod
    def from_organism_checkpoint(
        cls, checkpoint: OrganismCheckpoint, *, kernel_limits: KernelLimits
    ) -> CognitiveRuntimeCheckpoint:
        restored = cls.from_bytes(checkpoint.private_state, kernel_limits=kernel_limits)
        if restored.state.identity != checkpoint.identity:
            raise CognitiveRuntimeCheckpointError("envelope and private-state identities differ")
        return restored
