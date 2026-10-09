"""Longitudinal identity primitives owned by the Organism domain."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class OrganismIdentity:
    """An opaque, globally unique identity for one longitudinal Organism."""

    value: UUID

    @classmethod
    def create(cls, value: UUID) -> OrganismIdentity:
        """Wrap an explicitly supplied identity value.

        Identity generation is intentionally injected: this package does not
        draw from host entropy for state that can participate in causal life.
        The caller must use the applicable controlled identity/RNG contract.
        """
        return cls(value)

    @classmethod
    def parse(cls, value: str) -> OrganismIdentity:
        """Reconstitute an existing identity for an explicitly authorized continuation."""
        return cls(UUID(value))

    def __str__(self) -> str:
        return str(self.value)
