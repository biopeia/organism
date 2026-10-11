"""Canonical bounded Organism transition from opaque sense to checkpoint.

This operation composes the already-declared cognitive graph state only. It
does not compose agency, generative cognition, motor intent, or a complete
Organism dependency closure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from organism.checkpoint import OrganismCheckpoint
from organism.transition import (
    EmbodimentContextRef,
    OrganismLifecycle,
    OrganismTransitionError,
    OrganismTransitionProposal,
    OrganismTransitionState,
    WorldCausalCorrelation,
    accept_organism_transition,
)

from .composition import (
    CognitionComposition,
    EdgeLearningInstruction,
    compose_cognition_from_sense_signals,
)
from .generative.registry import GenerativeModelRegistry
from .limits import KernelLimits
from .portable_checkpoint import CognitiveRuntimeCheckpoint
from .sense_input import OpaqueSenseSignal, SenseChannelBinding, SenseProvenance


@dataclass(frozen=True, slots=True)
class AcceptedCognitionTransition:
    """Accepted canonical state and the staged cognition evidence for it."""

    state: OrganismTransitionState
    composition: CognitionComposition


def advance_organism_from_sense_signals(
    current: OrganismTransitionState,
    bindings: tuple[SenseChannelBinding, ...],
    signals: tuple[OpaqueSenseSignal, ...],
    *,
    expected_provenance: SenseProvenance,
    kernel_limits: KernelLimits,
    generative_registry: GenerativeModelRegistry | None = None,
    running_version: tuple[int, int, int] | None = None,
    learning: tuple[EdgeLearningInstruction, ...] = (),
) -> AcceptedCognitionTransition:
    """Advance one bounded cognition slice and accept it exactly once.

    The current opaque owner checkpoint must contain a valid
    ``CognitiveRuntimeCheckpoint``. Signal binding, graph composition, nested
    checkpoint creation, and the canonical ``organism_step`` increment are
    staged before returning. Any validation/restore/composition failure
    publishes no next ``OrganismTransitionState``.
    """
    if not isinstance(current, OrganismTransitionState):
        raise OrganismTransitionError("current must be OrganismTransitionState")
    if current.lifecycle is not OrganismLifecycle.ACTIVE:
        raise OrganismTransitionError("ordinary progression requires an ACTIVE Organism")
    if current.embodiment is None:
        raise OrganismTransitionError("sense-driven progression requires an explicit Embodiment context")
    if not isinstance(expected_provenance, SenseProvenance):
        raise OrganismTransitionError("expected_provenance must be a SenseProvenance")
    if not isinstance(kernel_limits, KernelLimits):
        raise OrganismTransitionError("kernel_limits must be explicit KernelLimits")

    _validate_embodiment_applicability(current.embodiment, expected_provenance)
    if (
        current.world_correlation is not None
        and current.world_correlation.world_id == expected_provenance.world_id
        and expected_provenance.causal_tick <= current.world_correlation.causal_tick
    ):
        raise OrganismTransitionError(
            "World causal correlation must advance within the same World"
        )

    runtime = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        current.owner_checkpoint,
        kernel_limits=kernel_limits,
        generative_registry=generative_registry,
        running_version=running_version,
    )
    composition = compose_cognition_from_sense_signals(
        runtime.state,
        runtime.graph,
        bindings,
        signals,
        expected_provenance=expected_provenance,
        learning=learning,
    )
    next_runtime = replace(
        runtime,
        state=composition.state,
        graph=composition.graph,
    )
    next_checkpoint = next_runtime.to_organism_checkpoint()
    proposal = OrganismTransitionProposal(
        owner_checkpoint=next_checkpoint,
        embodiment=current.embodiment,
        world_correlation=WorldCausalCorrelation(
            expected_provenance.world_id,
            expected_provenance.causal_tick,
        ),
    )
    accepted = accept_organism_transition(current, proposal)
    return AcceptedCognitionTransition(state=accepted, composition=composition)


def _validate_embodiment_applicability(
    embodiment: EmbodimentContextRef,
    provenance: SenseProvenance,
) -> None:
    if (
        embodiment.body_id != provenance.body_id
        or embodiment.embodiment_episode_id != provenance.embodiment_episode_id
        or embodiment.morphology_revision != provenance.morphology_revision
    ):
        raise OrganismTransitionError(
            "sense provenance does not match the current Embodiment context"
        )


__all__ = ["AcceptedCognitionTransition", "advance_organism_from_sense_signals"]
