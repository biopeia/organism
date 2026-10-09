"""Typed primitive for Organism experiential-evidence counts ($T_e$)."""

from __future__ import annotations


class EvidenceCount(int):
    """A non-negative integer count of observed evidence events.

    This count tracks empirical samples/evidence arrivals, not scheduler cycles,
    physical duration, or host wall time. It carries no evidence content or
    factual authority by itself.
    """

    def __new__(cls, value: int = 0) -> EvidenceCount:
        if not (type(value) is int or isinstance(value, cls)):
            raise TypeError("EvidenceCount requires a plain integer observation count")
        if value < 0:
            raise ValueError(f"EvidenceCount cannot be negative: {value}")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"EvidenceCount({int(self)})"


SampleCount = EvidenceCount

__all__ = ["EvidenceCount", "SampleCount"]
