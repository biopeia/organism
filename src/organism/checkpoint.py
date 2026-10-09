"""Opaque Organism-owned checkpoint boundary.

This module defines an in-memory contract, not a durable wire/storage format.
World may coordinate these checkpoints but does not interpret their private
payload.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable

from .identity import OrganismIdentity


class MissingCheckpointDependency(ValueError):
    """A referenced private-state artifact is unavailable."""


class CheckpointDependencyMismatch(ValueError):
    """A referenced artifact does not match its declared content digest."""


class InvalidOrganismCheckpoint(ValueError):
    """A serialized checkpoint is malformed or fails its integrity check."""


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

    def to_bytes(self) -> bytes:
        """Encode a deterministic, versioned portable envelope.

        The envelope transports owner-produced opaque state; it does not
        define how an Organism assembles or restores that state.
        """
        document = {
            "schema": "biopeia.organism-checkpoint",
            "schema_version": 1,
            "identity": str(self.identity),
            "private_state_base64": base64.b64encode(self.private_state).decode("ascii"),
            "private_state_sha256": sha256(self.private_state).hexdigest(),
            "dependencies": [
                {"locator": item.locator, "sha256": item.sha256_hex}
                for item in self.dependencies
            ],
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> OrganismCheckpoint:
        """Decode and integrity-check a versioned envelope; reject unknown shapes."""
        try:
            document = json.loads(payload)
            if not isinstance(document, dict) or set(document) != {
                "schema", "schema_version", "identity", "private_state_base64",
                "private_state_sha256", "dependencies",
            }:
                raise InvalidOrganismCheckpoint("checkpoint envelope has an invalid shape")
            if document["schema"] != "biopeia.organism-checkpoint" or document["schema_version"] != 1:
                raise InvalidOrganismCheckpoint("unsupported checkpoint schema or version")
            identity = OrganismIdentity.parse(document["identity"])
            state = base64.b64decode(document["private_state_base64"], validate=True)
            if sha256(state).hexdigest() != document["private_state_sha256"]:
                raise InvalidOrganismCheckpoint("private Organism state digest mismatch")
            raw_dependencies = document["dependencies"]
            if not isinstance(raw_dependencies, list):
                raise InvalidOrganismCheckpoint("dependencies must be a list")
            dependencies = tuple(
                CheckpointDependency(item["locator"], item["sha256"])
                for item in raw_dependencies
                if isinstance(item, dict) and set(item) == {"locator", "sha256"}
            )
            if len(dependencies) != len(raw_dependencies):
                raise InvalidOrganismCheckpoint("dependency entry has an invalid shape")
            return cls(identity, state, dependencies)
        except InvalidOrganismCheckpoint:
            raise
        except (TypeError, ValueError, KeyError, binascii.Error, json.JSONDecodeError) as exc:
            raise InvalidOrganismCheckpoint("checkpoint envelope is malformed") from exc
