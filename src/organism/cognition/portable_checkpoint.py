"""Bounded portable composition for extracted Organism-owned runtime state.

This composes graph structure/exact learned weights and eligibility state, recurrent activation
state, optional genetic state, and optionally a generative resident's durable
internal state. Model owners and transient evidence resolvers remain external.
It is not a complete cognition or Organism checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from organism.checkpoint import (
    OrganismCheckpoint,
)
from organism.genetics import GeneExpressionState, Genome, GermlineState
from organism.genetics.checkpoint import (
    export_expression,
    export_genome,
    export_germline,
    restore_expression,
    restore_genome,
    restore_germline,
)
from organism.genetics.schema import DEFAULT_GENOME_SCHEMA, GenomeSchema
from organism.identity import OrganismIdentity
from organism.memory import ExperienceRecord

from .activation import SensoryNormalizer
from .checkpoint import (
    export_graph_checkpoint,
    export_safety_state,
    export_sensory_normalizers,
    restore_graph_checkpoint,
    restore_safety_state,
    restore_sensory_normalizers,
)
from .generative.registry import GenerativeModelRegistry
from .generative.resident import ResidentGenerativeCognition
from .graph import CognitiveGraph, GraphError
from .limits import KernelLimits
from .metaplasticity import SafetyState
from .transition import CognitiveTransitionError, CognitiveTransitionState


class CognitiveRuntimeCheckpointError(ValueError):
    """Raised when a composed cognitive runtime checkpoint is invalid."""


def _reject_duplicate_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CognitiveRuntimeCheckpointError(
                f"duplicate JSON field in cognitive checkpoint: {key}"
            )
        result[key] = value
    return result


def _validate_genetic_state(
    genome: Genome | None,
    germline_state: GermlineState | None,
    expression_state: GeneExpressionState | None,
) -> None:
    genetic_values = (genome, germline_state, expression_state)
    if all(value is None for value in genetic_values):
        return
    if any(value is None for value in genetic_values):
        raise CognitiveRuntimeCheckpointError(
            "genome, germline_state, and expression_state must be supplied together"
        )
    if not isinstance(genome, Genome):
        raise CognitiveRuntimeCheckpointError("genome must be Genome or None")
    if not isinstance(germline_state, GermlineState):
        raise CognitiveRuntimeCheckpointError("germline_state must be GermlineState or None")
    if not isinstance(expression_state, GeneExpressionState):
        raise CognitiveRuntimeCheckpointError(
            "expression_state must be GeneExpressionState or None"
        )
    try:
        restored_germline = restore_germline(export_germline(germline_state), genome)
        restore_expression(
            export_expression(expression_state), genome, germline=restored_germline
        )
    except (TypeError, ValueError) as exc:
        raise CognitiveRuntimeCheckpointError(
            "genetic state is inconsistent with its genome and germline"
        ) from exc


@dataclass(frozen=True, slots=True)
class CognitiveRuntimeCheckpoint:
    """A bounded composition of Organism-owned cognitive and memory state.

    This payload includes only explicitly supplied state. The genome, germline
    and expression state are an all-or-none unit. A generative resident
    contributes its durable state only when it has no active external model
    registration or factual-evidence resolver. Those capabilities have no
    verified artifact binding and cannot be declared portable here. This is
    not a complete Organism checkpoint.
    """

    state: CognitiveTransitionState
    graph: CognitiveGraph
    experience_records: tuple[ExperienceRecord, ...] = ()
    safety_state: SafetyState | None = None
    sensory_normalizers: dict[str, SensoryNormalizer] = field(default_factory=dict)
    generative_resident: ResidentGenerativeCognition | None = None
    genome: Genome | None = None
    germline_state: GermlineState | None = None
    expression_state: GeneExpressionState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, CognitiveTransitionState):
            raise CognitiveRuntimeCheckpointError("state must be CognitiveTransitionState")
        if not isinstance(self.graph, CognitiveGraph):
            raise CognitiveRuntimeCheckpointError("graph must be CognitiveGraph")
        if self.safety_state is not None and not isinstance(self.safety_state, SafetyState):
            raise CognitiveRuntimeCheckpointError("safety_state must be SafetyState or None")
        if self.generative_resident is not None and not isinstance(
            self.generative_resident, ResidentGenerativeCognition
        ):
            raise CognitiveRuntimeCheckpointError(
                "generative_resident must be ResidentGenerativeCognition or None"
            )
        if self.generative_resident is not None and self.generative_resident.organism_id != str(
            self.state.identity
        ):
            raise CognitiveRuntimeCheckpointError(
                "generative resident belongs to a different Organism identity"
            )
        genetic_values = (self.genome, self.germline_state, self.expression_state)
        if any(value is not None for value in genetic_values) and not all(
            value is not None for value in genetic_values
        ):
            raise CognitiveRuntimeCheckpointError(
                "genome, germline_state, and expression_state must be supplied together"
            )
        if self.genome is not None and not isinstance(self.genome, Genome):
            raise CognitiveRuntimeCheckpointError("genome must be Genome or None")
        if self.germline_state is not None and not isinstance(self.germline_state, GermlineState):
            raise CognitiveRuntimeCheckpointError("germline_state must be GermlineState or None")
        if self.expression_state is not None and not isinstance(
            self.expression_state, GeneExpressionState
        ):
            raise CognitiveRuntimeCheckpointError(
                "expression_state must be GeneExpressionState or None"
            )
        _validate_genetic_state(self.genome, self.germline_state, self.expression_state)
        if not isinstance(self.sensory_normalizers, dict) or any(
            not isinstance(key, str) or not isinstance(value, SensoryNormalizer)
            for key, value in self.sensory_normalizers.items()
        ):
            raise CognitiveRuntimeCheckpointError(
                "sensory_normalizers must map string IDs to SensoryNormalizer values"
            )
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
        # Nested germline/expression values are mutable; revalidate the actual
        # snapshot being serialized instead of trusting construction-time state.
        _validate_genetic_state(self.genome, self.germline_state, self.expression_state)
        if (
            self.generative_resident is not None
            and self.generative_resident.has_external_runtime_dependencies
        ):
            raise CognitiveRuntimeCheckpointError(
                "portable checkpoint does not support external generative dependencies"
            )
        payload = {
            "schema": "biopeia.organism.cognitive-runtime",
            "schema_version": 9,
            # This owner payload supports only the self-contained subset.
            # External model objects and factual resolvers are rejected above.
            "dependency_closure": [],
            "kernel_limits": self.graph.kernel_limits.checkpoint_configuration(),
            "identity": str(self.state.identity),
            "transition": self.state.checkpoint_payload(),
            "graph": export_graph_checkpoint(self.graph),
            "experience_records": [
                record.canonical_payload() for record in self.experience_records
            ],
            "safety_state": (
                None if self.safety_state is None else export_safety_state(self.safety_state)
            ),
            "sensory_normalizers": export_sensory_normalizers(self.sensory_normalizers),
            "generative_resident": (
                None
                if self.generative_resident is None
                else self.generative_resident.checkpoint()
            ),
            "genome": None if self.genome is None else export_genome(self.genome),
            "germline_state": (
                None if self.germline_state is None else export_germline(self.germline_state)
            ),
            "expression_state": (
                None if self.expression_state is None else export_expression(self.expression_state)
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        kernel_limits: KernelLimits,
        generative_registry: GenerativeModelRegistry | None = None,
        running_version: tuple[int, int, int] | None = None,
        genome_schema: GenomeSchema = DEFAULT_GENOME_SCHEMA,
    ) -> CognitiveRuntimeCheckpoint:
        if not isinstance(kernel_limits, KernelLimits):
            raise CognitiveRuntimeCheckpointError("kernel_limits must be explicit KernelLimits")
        if not isinstance(payload, bytes):
            raise CognitiveRuntimeCheckpointError("cognitive checkpoint payload must be bytes")
        if len(payload) > kernel_limits.max_owner_checkpoint_bytes:
            raise CognitiveRuntimeCheckpointError("cognitive checkpoint exceeds byte limit")
        try:
            document = json.loads(payload, object_pairs_hook=_reject_duplicate_json_fields)
            if not isinstance(document, dict) or set(document) != {
                "schema",
                "schema_version",
                "identity",
                "transition",
                "graph",
                "experience_records",
                "safety_state",
                "sensory_normalizers",
                "generative_resident",
                "genome",
                "germline_state",
                "expression_state",
                "dependency_closure",
                "kernel_limits",
            }:
                raise CognitiveRuntimeCheckpointError("invalid cognitive runtime checkpoint shape")
            if (
                document["schema"] != "biopeia.organism.cognitive-runtime"
                or type(document["schema_version"]) is not int
                or document["schema_version"] != 9
            ):
                raise CognitiveRuntimeCheckpointError("unsupported cognitive runtime checkpoint")
            if not kernel_limits.matches_checkpoint_configuration(document["kernel_limits"]):
                raise CognitiveRuntimeCheckpointError(
                    "kernel limits do not match the checkpoint configuration"
                )
            if document["dependency_closure"] != []:
                raise CognitiveRuntimeCheckpointError(
                    "unsupported external dependency closure in cognitive checkpoint"
                )
            if (
                document["generative_resident"] is not None
                and generative_registry is not None
                and generative_registry.model_ids
            ):
                raise CognitiveRuntimeCheckpointError(
                    "portable checkpoint does not support external generative dependencies"
                )
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
            raw_safety = document["safety_state"]
            safety_state = None if raw_safety is None else restore_safety_state(raw_safety)
            normalizers = restore_sensory_normalizers(document["sensory_normalizers"])
            raw_generative = document["generative_resident"]
            generative_resident = (
                None
                if raw_generative is None
                else ResidentGenerativeCognition.from_checkpoint(
                    raw_generative,
                    organism_id=str(identity),
                    registry=generative_registry,
                )
            )
            if (
                generative_resident is not None
                and generative_resident.has_external_runtime_dependencies
            ):
                raise CognitiveRuntimeCheckpointError(
                    "portable checkpoint contains external generative dependencies"
                )
            raw_genome = document["genome"]
            raw_germline = document["germline_state"]
            raw_expression = document["expression_state"]
            genetic_values = (raw_genome, raw_germline, raw_expression)
            if all(value is None for value in genetic_values):
                genome = None
                germline_state = None
                expression_state = None
            elif any(value is None for value in genetic_values):
                raise CognitiveRuntimeCheckpointError(
                    "genome, germline_state, and expression_state must be present together"
                )
            else:
                if running_version is None:
                    raise CognitiveRuntimeCheckpointError(
                        "running_version is required to restore genetic state"
                    )
                if not isinstance(raw_genome, dict):
                    raise CognitiveRuntimeCheckpointError("genome state must be an object")
                genome = restore_genome(
                    raw_genome,
                    kernel_limits=kernel_limits,
                    running_version=running_version,
                    schema=genome_schema,
                )
                if not isinstance(raw_germline, dict) or not isinstance(raw_expression, dict):
                    raise CognitiveRuntimeCheckpointError(
                        "germline and expression states must be objects"
                    )
                germline_state = restore_germline(raw_germline, genome, schema=genome_schema)
                expression_state = restore_expression(
                    raw_expression, genome, germline=germline_state
                )
            return cls(
                state=state,
                graph=graph,
                experience_records=records,
                safety_state=safety_state,
                sensory_normalizers=normalizers,
                generative_resident=generative_resident,
                genome=genome,
                germline_state=germline_state,
                expression_state=expression_state,
            )
        except CognitiveRuntimeCheckpointError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, GraphError,
                CognitiveTransitionError) as exc:
            raise CognitiveRuntimeCheckpointError("malformed cognitive runtime checkpoint") from exc

    def to_organism_checkpoint(self) -> OrganismCheckpoint:
        """Wrap the supported self-contained owner payload without caller refs."""
        return OrganismCheckpoint(self.state.identity, self.to_bytes(), ())

    @classmethod
    def from_organism_checkpoint(
        cls,
        checkpoint: OrganismCheckpoint,
        *,
        kernel_limits: KernelLimits,
        generative_registry: GenerativeModelRegistry | None = None,
        running_version: tuple[int, int, int] | None = None,
        genome_schema: GenomeSchema = DEFAULT_GENOME_SCHEMA,
    ) -> CognitiveRuntimeCheckpoint:
        """Restore only this owner's explicitly self-contained payload subset."""
        if checkpoint.dependencies:
            raise CognitiveRuntimeCheckpointError(
                "envelope dependencies do not match the supported owner payload"
            )
        restored = cls.from_bytes(
            checkpoint.private_state,
            kernel_limits=kernel_limits,
            generative_registry=generative_registry,
            running_version=running_version,
            genome_schema=genome_schema,
        )
        if restored.state.identity != checkpoint.identity:
            raise CognitiveRuntimeCheckpointError("envelope and private-state identities differ")
        return restored
