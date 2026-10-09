"""Foundational organism conventions independent of runtime composition."""

from .epistemic import DEFAULT_EPISTEMIC_CONVENTIONS, EpistemicConventions, RecencyClass
from .limits import OrganismLimits

__all__ = [
    "DEFAULT_EPISTEMIC_CONVENTIONS",
    "EpistemicConventions",
    "OrganismLimits",
    "RecencyClass",
]
