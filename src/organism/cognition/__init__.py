"""Cognitive graph, genome and plasticity kernel (roadmap v0.55+).

This package never imports lab -- cognition must stay
independent of the evaluation apparatus that will eventually score it
(see tests/experimental_integrity/test_ground_truth_boundary.py).
"""

from .learning import ShadowPrediction
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
    "CognitiveTransitionError",
    "CognitiveTransitionState",
    "CognitiveRuntimeCheckpoint",
    "CognitiveRuntimeCheckpointError",
    "ShadowPrediction",
    "advance_cognition",
]
