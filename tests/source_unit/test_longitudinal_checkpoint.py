from __future__ import annotations

import json
from hashlib import sha256
from uuid import UUID

import pytest

from organism import (
    CheckpointDependency,
    CheckpointDependencyMismatch,
    InvalidLongitudinalCheckpoint,
    OrganismIdentity,
    OrganismLongitudinalCheckpoint,
)
from organism.checkpoint import REQUIRED_LONGITUDINAL_COMPONENTS


def _checkpoint() -> OrganismLongitudinalCheckpoint:
    return OrganismLongitudinalCheckpoint(
        identity=OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")),
        components={name: f"{name}:payload".encode() for name in REQUIRED_LONGITUDINAL_COMPONENTS},
    )


def test_longitudinal_checkpoint_round_trips_exact_component_bytes() -> None:
    checkpoint = _checkpoint()
    restored = OrganismLongitudinalCheckpoint.from_bytes(checkpoint.to_bytes())

    assert restored.identity == checkpoint.identity
    assert dict(restored.components) == dict(checkpoint.components)
    assert restored.to_bytes() == checkpoint.to_bytes()
    assert OrganismLongitudinalCheckpoint.from_organism_checkpoint(
        checkpoint.to_organism_checkpoint()
    ).to_bytes() == checkpoint.to_bytes()


def test_longitudinal_checkpoint_rejects_missing_or_unknown_components() -> None:
    components = dict(_checkpoint().components)
    components.pop("agency")
    with pytest.raises(InvalidLongitudinalCheckpoint, match="closure mismatch"):
        OrganismLongitudinalCheckpoint(
            OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")), components
        )
    components["agency"] = b"agency"
    components["unregistered"] = b"must not be silently ignored"
    with pytest.raises(InvalidLongitudinalCheckpoint, match="closure mismatch"):
        OrganismLongitudinalCheckpoint(
            OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")), components
        )


def test_capture_uses_a_complete_identity_bound_encoder_registry() -> None:
    identity = OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041"))
    states = {name: name for name in REQUIRED_LONGITUDINAL_COMPONENTS}
    encoders = {
        name: lambda value, owner: f"{owner}:{value}".encode()
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    checkpoint = OrganismLongitudinalCheckpoint.capture_components(identity, states, encoders)

    assert all(str(identity).encode() in payload for payload in checkpoint.components.values())
    with pytest.raises(InvalidLongitudinalCheckpoint, match="encoder registry"):
        OrganismLongitudinalCheckpoint.capture_components(
            identity,
            states,
            {key: value for key, value in encoders.items() if key != "agency"},
        )

def test_longitudinal_checkpoint_rejects_tampered_component_digest() -> None:
    document = json.loads(_checkpoint().to_bytes())
    document["components"]["cognitive_graph"]["payload_base64"] = "Y29ycnVwdGVk"

    with pytest.raises(InvalidLongitudinalCheckpoint, match="digest mismatch"):
        OrganismLongitudinalCheckpoint.from_bytes(
            json.dumps(document, separators=(",", ":")).encode()
        )


def test_longitudinal_checkpoint_rejects_duplicate_json_keys() -> None:
    payload = _checkpoint().to_bytes()
    duplicate = payload[:-1] + b',"schema_version":1}'
    with pytest.raises(InvalidLongitudinalCheckpoint, match="malformed"):
        OrganismLongitudinalCheckpoint.from_bytes(duplicate)


def test_restore_requires_complete_decoders_and_dependency_closure() -> None:
    checkpoint = _checkpoint()
    decoders = {
        name: lambda payload, identity: payload.decode()
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    with pytest.raises(InvalidLongitudinalCheckpoint, match="decoder registry"):
        checkpoint.restore_components(
            {key: value for key, value in decoders.items() if key != "agency"},
            read_dependency=lambda locator: b"artifact",
        )

    dependency = b"model weights"
    checkpoint = OrganismLongitudinalCheckpoint(
        identity=checkpoint.identity,
        components=checkpoint.components,
        dependencies=(CheckpointDependency("model://predictor", sha256(dependency).hexdigest()),),
    )

    def fail_decode(payload: bytes, identity: OrganismIdentity) -> object:
        raise ValueError("invalid subsystem payload")

    with pytest.raises(InvalidLongitudinalCheckpoint, match="failed to restore"):
        checkpoint.restore_components(
            {**decoders, "agency": fail_decode},
            read_dependency=lambda locator: dependency,
        )
    with pytest.raises(CheckpointDependencyMismatch, match="model://predictor"):
        checkpoint.restore_components(decoders, read_dependency=lambda locator: b"wrong")


def test_longitudinal_checkpoint_captures_and_restores_action_domain() -> None:
    from organism.actuation.domain import ActionDomain

    identity = OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041"))
    domain = ActionDomain()
    states = {name: name for name in REQUIRED_LONGITUDINAL_COMPONENTS}
    states["action_commitments"] = domain

    encoders = {
        name: (lambda value, owner: f"{owner}:{value}".encode())
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    encoders["action_commitments"] = lambda value, owner: value.to_bytes()

    checkpoint = OrganismLongitudinalCheckpoint.capture_components(identity, states, encoders)
    assert "action_commitments" in checkpoint.components

    decoders = {
        name: (lambda payload, owner: payload.decode())
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    decoders["action_commitments"] = lambda payload, owner: ActionDomain.from_bytes(payload)

    restored_components = checkpoint.restore_components(
        decoders, read_dependency=lambda locator: b""
    )
    restored_domain = restored_components["action_commitments"]
    assert isinstance(restored_domain, ActionDomain)
    assert restored_domain.checkpoint() == domain.checkpoint()

