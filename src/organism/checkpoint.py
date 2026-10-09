"""Opaque Organism-owned checkpoint boundary.

This module defines an in-memory contract, not a durable wire/storage format.
World may coordinate these checkpoints but does not interpret their private
payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable

from .identity import OrganismIdentity


class MissingCheckpointDependency(ValueError):
    """A referenced private-state artifact is unavailable."""


class CheckpointDependencyMismatch(ValueError):
    """A referenced artifact does not match its declared content digest."""


@dataclass(frozen=True, slots=True)
class CheckpointDependency:
    """A durable content reference required to restore private Organism state."""

    locator: str
    sha256_hex: str

    def __post_init__(self) -> None:
        if not self.locator.strip():
            raise ValueError("dependency locator must be non-empty")
        if len(self.sha256_hex) != 64 or any(c not in "0123456789abcdef" for c in self.sha256_hex):
            raise ValueError("sha256_hex must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class OrganismCheckpoint:
    """Opaque owner-produced state plus the references required to restore it.

    ``private_state`` is intentionally uninterpreted outside Organism. A
    checkpoint is only dependency-complete when every referenced object is
    available and matches its content digest.
    """

    identity: OrganismIdentity
    private_state: bytes
    dependencies: tuple[CheckpointDependency, ...] = ()

    def verify_dependency_closure(self, read_bytes: Callable[[str], bytes]) -> None:
        """Raise if any declared dependency is missing or has changed."""
        for dependency in self.dependencies:
            try:
                payload = read_bytes(dependency.locator)
            except (FileNotFoundError, KeyError) as exc:
                raise MissingCheckpointDependency(dependency.locator) from exc
            actual = sha256(payload).hexdigest()
            if actual != dependency.sha256_hex:
                raise CheckpointDependencyMismatch(dependency.locator)
