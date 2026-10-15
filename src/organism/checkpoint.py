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
from types import MappingProxyType
from typing import Callable, Mapping

from .identity import OrganismIdentity


class MissingCheckpointDependency(ValueError):
    """A referenced private-state artifact is unavailable."""


class CheckpointDependencyMismatch(ValueError):
    """A referenced artifact does not match its declared content digest."""


class InvalidOrganismCheckpoint(ValueError):
    """A serialized checkpoint is malformed or fails its integrity check."""


class InvalidLongitudinalCheckpoint(ValueError):
    """A longitudinal checkpoint omits or corrupts declared owner state."""


# These are the payload sections required by the currently declared
# continuation-closure taxonomy. Component codecs remain owned by their
# respective Organism subsystems; this aggregate refuses a partial profile.
REQUIRED_LONGITUDINAL_COMPONENTS = frozenset({
    "lifecycle",
    "cognitive_transition",
    "cognitive_graph",
    "experience_memory",
    "genetic_expression",
    "learning_safety",
    "sensory_normalizers",
    "agency",
    "competencies",
    "action_commitments",
})


@dataclass(frozen=True, slots=True)
class OrganismLongitudinalCheckpoint:
    """Organism-owned aggregate of exact, subsystem-encoded continuation state.

    Component payloads are opaque here so each subsystem retains codec
    ownership. The aggregate enforces the declared closure and integrity; it
    intentionally does not claim that a caller has materialized runtime
    objects or resolved external dependencies.
    """

    identity: OrganismIdentity
    components: Mapping[str, bytes]
    dependencies: tuple[CheckpointDependency, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.identity, OrganismIdentity):
            raise InvalidLongitudinalCheckpoint("identity must be an OrganismIdentity")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise InvalidLongitudinalCheckpoint("unsupported longitudinal checkpoint version")
        if not isinstance(self.components, Mapping):
            raise InvalidLongitudinalCheckpoint("components must be a mapping")
        if any(not isinstance(name, str) for name in self.components):
            raise InvalidLongitudinalCheckpoint("component names must be strings")
        if set(self.components) != REQUIRED_LONGITUDINAL_COMPONENTS:
            missing = sorted(REQUIRED_LONGITUDINAL_COMPONENTS - set(self.components))
            extra = sorted(set(self.components) - REQUIRED_LONGITUDINAL_COMPONENTS)
            raise InvalidLongitudinalCheckpoint(
                f"longitudinal component closure mismatch (missing={missing}, extra={extra})"
            )
        if not isinstance(self.dependencies, tuple) or any(
            not isinstance(item, CheckpointDependency) for item in self.dependencies
        ):
            raise InvalidLongitudinalCheckpoint("dependencies must be checkpoint dependencies")
        if len({item.locator for item in self.dependencies}) != len(self.dependencies):
            raise InvalidLongitudinalCheckpoint("duplicate external dependency locator")
        normalized: dict[str, bytes] = {}
        for name, payload in self.components.items():
            if not isinstance(name, str) or not isinstance(payload, bytes) or not payload:
                raise InvalidLongitudinalCheckpoint("component payloads must be non-empty bytes")
            normalized[name] = bytes(payload)
        object.__setattr__(self, "components", MappingProxyType(normalized))

    def to_bytes(self) -> bytes:
        document = {
            "schema": "biopeia.organism-longitudinal-checkpoint",
            "schema_version": self.schema_version,
            "identity": str(self.identity),
            "components": {
                name: {
                    "payload_base64": base64.b64encode(payload).decode("ascii"),
                    "sha256": sha256(payload).hexdigest(),
                }
                for name, payload in sorted(self.components.items())
            },
            "dependencies": [
                {"locator": item.locator, "sha256": item.sha256_hex}
                for item in self.dependencies
            ],
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OrganismLongitudinalCheckpoint":
        try:
            document = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
            expected = {"schema", "schema_version", "identity", "components", "dependencies"}
            if not isinstance(document, dict) or set(document) != expected:
                raise InvalidLongitudinalCheckpoint("longitudinal checkpoint has an invalid shape")
            if (
                document["schema"] != "biopeia.organism-longitudinal-checkpoint"
                or type(document["schema_version"]) is not int
                or document["schema_version"] != 1
            ):
                raise InvalidLongitudinalCheckpoint("unsupported longitudinal checkpoint schema")
            raw_components = document["components"]
            if (
                not isinstance(raw_components, dict)
                or set(raw_components) != REQUIRED_LONGITUDINAL_COMPONENTS
            ):
                raise InvalidLongitudinalCheckpoint(
                    "longitudinal checkpoint component closure mismatch"
                )
            components: dict[str, bytes] = {}
            for name, entry in raw_components.items():
                if not isinstance(entry, dict) or set(entry) != {"payload_base64", "sha256"}:
                    raise InvalidLongitudinalCheckpoint(f"invalid {name} component envelope")
                if not isinstance(entry["payload_base64"], str) or not isinstance(
                    entry["sha256"], str
                ):
                    raise InvalidLongitudinalCheckpoint(f"invalid {name} component envelope")
                component = base64.b64decode(entry["payload_base64"], validate=True)
                if not component or sha256(component).hexdigest() != entry["sha256"]:
                    raise InvalidLongitudinalCheckpoint(f"{name} component digest mismatch")
                components[name] = component
            raw_dependencies = document["dependencies"]
            if not isinstance(raw_dependencies, list):
                raise InvalidLongitudinalCheckpoint("dependencies must be a list")
            dependencies = tuple(
                CheckpointDependency(item["locator"], item["sha256"])
                for item in raw_dependencies
                if isinstance(item, dict) and set(item) == {"locator", "sha256"}
            )
            if len(dependencies) != len(raw_dependencies):
                raise InvalidLongitudinalCheckpoint("invalid dependency entry")
            return cls(OrganismIdentity.parse(document["identity"]), components, dependencies)
        except InvalidLongitudinalCheckpoint:
            raise
        except (TypeError, ValueError, KeyError, binascii.Error, json.JSONDecodeError) as exc:
            raise InvalidLongitudinalCheckpoint("longitudinal checkpoint is malformed") from exc

    def verify_dependency_closure(self, read_bytes: Callable[[str], bytes]) -> None:
        for dependency in self.dependencies:
            try:
                value = read_bytes(dependency.locator)
            except (FileNotFoundError, KeyError) as exc:
                raise MissingCheckpointDependency(dependency.locator) from exc
            if sha256(value).hexdigest() != dependency.sha256_hex:
                raise CheckpointDependencyMismatch(dependency.locator)

    def restore_components(
        self,
        decoders: Mapping[str, Callable[[bytes], object]],
        *,
        read_dependency: Callable[[str], bytes],
    ) -> Mapping[str, object]:
        """Decode the full profile only after verifying its external closure.

        Decoder callbacks are explicitly supplied by the owning runtime because
        subsystem restoration may require caller-owned configuration. A partial
        decoder registry, a decoder failure, or an unresolved external artifact
        fails the whole restoration rather than returning a partial profile.
        Callbacks remain responsible for validating their subsystem-specific
        identity, configuration and applicability invariants.
        """
        if not isinstance(decoders, Mapping) or set(decoders) != REQUIRED_LONGITUDINAL_COMPONENTS:
            raise InvalidLongitudinalCheckpoint("decoder registry does not match component closure")
        if any(not callable(decoder) for decoder in decoders.values()):
            raise InvalidLongitudinalCheckpoint("every component decoder must be callable")
        self.verify_dependency_closure(read_dependency)
        restored: dict[str, object] = {}
        for name in sorted(REQUIRED_LONGITUDINAL_COMPONENTS):
            try:
                restored[name] = decoders[name](self.components[name])
            except Exception as exc:
                raise InvalidLongitudinalCheckpoint(
                    f"failed to restore longitudinal component: {name}"
                ) from exc
        return MappingProxyType(restored)

    def to_organism_checkpoint(self) -> OrganismCheckpoint:
        """Wrap this aggregate in the existing opaque owner checkpoint envelope."""
        return OrganismCheckpoint(
            identity=self.identity,
            private_state=self.to_bytes(),
            dependencies=self.dependencies,
        )

    @classmethod
    def from_organism_checkpoint(
        cls, checkpoint: OrganismCheckpoint
    ) -> "OrganismLongitudinalCheckpoint":
        """Materialize the aggregate only when its identity and references agree."""
        if not isinstance(checkpoint, OrganismCheckpoint):
            raise InvalidLongitudinalCheckpoint("expected an OrganismCheckpoint envelope")
        restored = cls.from_bytes(checkpoint.private_state)
        if restored.identity != checkpoint.identity:
            raise InvalidLongitudinalCheckpoint("longitudinal checkpoint identity mismatch")
        if restored.dependencies != checkpoint.dependencies:
            raise InvalidLongitudinalCheckpoint("longitudinal dependency closure mismatch")
        return restored


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate checkpoint field: {key}")
        document[key] = value
    return document


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
            document = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
            if not isinstance(document, dict) or set(document) != {
                "schema", "schema_version", "identity", "private_state_base64",
                "private_state_sha256", "dependencies",
            }:
                raise InvalidOrganismCheckpoint("checkpoint envelope has an invalid shape")
            schema_version = document["schema_version"]
            if (
                document["schema"] != "biopeia.organism-checkpoint"
                or type(schema_version) is not int
                or schema_version != 1
            ):
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
