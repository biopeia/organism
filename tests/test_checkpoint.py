import base64
import json
from hashlib import sha256
from uuid import UUID

import pytest

from organism import (
    CheckpointDependency,
    CheckpointDependencyMismatch,
    InvalidOrganismCheckpoint,
    MissingCheckpointDependency,
    OrganismCheckpoint,
    OrganismIdentity,
)


def checkpoint(payload: bytes = b"private-model-weights") -> tuple[OrganismCheckpoint, bytes]:
    ref = CheckpointDependency(
        locator="sha256/weights-1",
        sha256_hex=sha256(payload).hexdigest(),
    )
    identity = OrganismIdentity.create(UUID("9fe6ebc6-a82f-4d41-9f65-5e9707a29b05"))
    return OrganismCheckpoint(identity, b"opaque-organism-state", (ref,)), payload


def test_checkpoint_verifies_dependency_closure() -> None:
    saved, payload = checkpoint()

    saved.verify_dependency_closure(lambda locator: payload)


def test_missing_dependency_means_checkpoint_is_not_complete() -> None:
    saved, _ = checkpoint()

    with pytest.raises(MissingCheckpointDependency):
        saved.verify_dependency_closure(lambda locator: (_ for _ in ()).throw(FileNotFoundError(locator)))


def test_changed_dependency_is_rejected() -> None:
    saved, _ = checkpoint()

    with pytest.raises(CheckpointDependencyMismatch):
        saved.verify_dependency_closure(lambda locator: b"different artifact")


def test_dependency_requires_valid_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        CheckpointDependency("artifact", "not-a-digest")


def test_portable_envelope_round_trips_opaque_longitudinal_state() -> None:
    saved, payload = checkpoint()

    restored = OrganismCheckpoint.from_bytes(saved.to_bytes())

    assert restored == saved
    restored.verify_dependency_closure(lambda locator: payload)


def test_portable_envelope_rejects_private_state_corruption() -> None:
    saved, _ = checkpoint()
    document = json.loads(saved.to_bytes())
    document["private_state_base64"] = base64.b64encode(b"changed-organism-state").decode()
    envelope = json.dumps(document).encode()

    with pytest.raises(InvalidOrganismCheckpoint, match="digest mismatch"):
        OrganismCheckpoint.from_bytes(envelope)


def test_portable_envelope_rejects_unknown_version_and_extra_fields() -> None:
    saved, _ = checkpoint()
    document = json.loads(saved.to_bytes())
    document["schema_version"] = 2
    with pytest.raises(InvalidOrganismCheckpoint, match="unsupported"):
        OrganismCheckpoint.from_bytes(json.dumps(document).encode())

    document["schema_version"] = 1
    document["unreviewed"] = True
    with pytest.raises(InvalidOrganismCheckpoint, match="invalid shape"):
        OrganismCheckpoint.from_bytes(json.dumps(document).encode())
