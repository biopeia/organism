"""Cognitive graph, genome and plasticity kernel (roadmap v0.55+).

This package never imports lab -- cognition must stay
independent of the evaluation apparatus that will eventually score it
(see tests/experimental_integrity/test_ground_truth_boundary.py).
"""

from .composition import (
    CognitionComposition,
    CognitiveStage,
    EdgeLearningInstruction,
    StageResult,
    compose_cognition_transition,
    compose_cognition_from_sense_signals,
)
from .learning import ShadowPrediction
from .sense_input import (
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    advance_cognition_from_sense_signals,
    bind_opaque_sense_signals,
)
from .owner_transition import (
    AcceptedCognitionTransition,
    advance_organism_from_sense_signals,
)
from .portable_checkpoint import (
    CognitiveRuntimeCheckpoint,
    CognitiveRuntimeCheckpointError,
)
from .transition import (
    CognitiveTransitionError,
    CognitiveTransitionState,
    advance_cognition,
)

__all__ = [
    "CognitionComposition",
    "CognitiveRuntimeCheckpoint",
    "CognitiveRuntimeCheckpointError",
    "AcceptedCognitionTransition",
    "CognitiveStage",
    "CognitiveTransitionError",
    "CognitiveTransitionState",
    "EdgeLearningInstruction",
    "OpaqueSenseSignal",
    "SenseChannelBinding",
    "SenseProvenance",
    "ShadowPrediction",
    "StageResult",
    "advance_cognition",
    "advance_cognition_from_sense_signals",
    "advance_organism_from_sense_signals",
    "bind_opaque_sense_signals",
    "compose_cognition_from_sense_signals",
    "compose_cognition_transition",
]
