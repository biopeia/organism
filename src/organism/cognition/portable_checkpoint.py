"""Bounded portable composition for the extracted cognitive graph runtime.

This composes graph structure/quantized learned weights with recurrent
activation state. It is not a complete cognition or Organism checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from organism.checkpoint import (
    CheckpointDependency,
    CheckpointDependencyMismatch,
    MissingCheckpointDependency,
    OrganismCheckpoint,
)
from organism.identity import OrganismIdentity
from organism.memory import ExperienceRecord

from .checkpoint import export_graph_checkpoint, restore_graph_checkpoint
from .graph import CognitiveGraph, GraphError
from .limits import KernelLimits
from .transition import CognitiveTransitionError, CognitiveTransitionState


class CognitiveRuntimeCheckpointError(ValueError):
    """Raised when a composed cognitive runtime checkpoint is invalid."""


@dataclass(frozen=True, slots=True)
class CognitiveRuntimeCheckpoint:
    """A bounded composition of Organism-owned cognitive and memory state.

    This payload includes only the graph, recurrent state, and explicitly
    supplied experience records. It is not a complete Organism checkpoint.
    """

    state: CognitiveTransitionState
    graph: CognitiveGraph
    experience_records: tuple[ExperienceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, CognitiveTransitionState):
            raise CognitiveRuntimeCheckpointError("state must be CognitiveTransitionState")
        if not isinstance(self.graph, CognitiveGraph):
            raise CognitiveRuntimeCheckpointError("graph must be CognitiveGraph")
        if not isinstance(self.experience_records, tuple) or any(
            not isinstance(record, ExperienceRecord) for record in self.experience_records
        ):
            raise CognitiveRuntimeCheckpointError(
                "experience_records must be a tuple of ExperienceRecord values"
            )
        record_ids: set[str] = set()
        expected_identity = str(self.state.identity)
        for record in self.experience_records:
            if record.organism_id != expected_identity:
                raise CognitiveRuntimeCheckpointError(
                    "experience record belongs to a different Organism identity"
                )
            if record.record_id in record_ids:
                raise CognitiveRuntimeCheckpointError("duplicate experience record identity")
            record_ids.add(record.record_id)

    def to_bytes(self) -> bytes:
        payload = {
            "schema": "biopeia.organism.cognitive-runtime",
            "schema_version": 2,
            "identity": str(self.state.identity),
            "transition": self.state.checkpoint_payload(),
            "graph": export_graph_checkpoint(self.graph),
            "experience_records": [
                record.canonical_payload() for record in self.experience_records
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(
        cls, payload: bytes, *, kernel_limits: KernelLimits
    ) -> CognitiveRuntimeCheckpoint:
        try:
            document = json.loads(payload)
            if not isinstance(document, dict) or set(document) != {
                "schema",
                "schema_version",
                "identity",
                "transition",
                "graph",
                "experience_records",
            }:
                raise CognitiveRuntimeCheckpointError("invalid cognitive runtime checkpoint shape")
            if document["schema"] != "biopeia.organism.cognitive-runtime" or document[
                "schema_version"
            ] != 2:
                raise CognitiveRuntimeCheckpointError("unsupported cognitive runtime checkpoint")
            identity = OrganismIdentity.parse(document["identity"])
            state = CognitiveTransitionState.from_checkpoint_payload(document["transition"])
            if state.identity != identity:
                raise CognitiveRuntimeCheckpointError("identity mismatch in cognitive checkpoint")
            graph = restore_graph_checkpoint(document["graph"], kernel_limits=kernel_limits)
            if graph is None:
                raise CognitiveRuntimeCheckpointError("graph is required")
            raw_records = document["experience_records"]
            if not isinstance(raw_records, list):
                raise CognitiveRuntimeCheckpointError("experience_records must be a list")
            records = tuple(ExperienceRecord.restore(item) for item in raw_records)
            return cls(state=state, graph=graph, experience_records=records)
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
        cls,
        checkpoint: OrganismCheckpoint,
        *,
        kernel_limits: KernelLimits,
        read_dependency: Callable[[str], bytes] | None = None,
    ) -> CognitiveRuntimeCheckpoint:
        """Restore the composed payload after validating declared dependencies.

        Dependency references are part of the checkpoint's portability closure.
        A caller restoring an envelope with references must provide a resolver;
        silently ignoring those references could accept an incomplete
        continuation.
        """
        if checkpoint.dependencies:
            if read_dependency is None:
                raise CognitiveRuntimeCheckpointError(
                    "dependency resolver is required for referenced Organism state"
                )
            try:
                checkpoint.verify_dependency_closure(read_dependency)
            except (MissingCheckpointDependency, CheckpointDependencyMismatch) as exc:
                raise CognitiveRuntimeCheckpointError(
                    "Organism checkpoint dependency closure is invalid"
                ) from exc
        restored = cls.from_bytes(checkpoint.private_state, kernel_limits=kernel_limits)
        if restored.state.identity != checkpoint.identity:
            raise CognitiveRuntimeCheckpointError("envelope and private-state identities differ")
        return restored
