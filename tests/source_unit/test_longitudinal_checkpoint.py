from __future__ import annotations

import json
from uuid import UUID

import pytest

from organism import (
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
        OrganismLongitudinalCheckpoint(OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")), components)

    components["agency_actuation"] = b"agency"
    components["unregistered"] = b"must not be silently ignored"
    with pytest.raises(InvalidLongitudinalCheckpoint, match="closure mismatch"):
        OrganismLongitudinalCheckpoint(OrganismIdentity(UUID("00000000-0000-0000-0000-000000000041")), components)


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
