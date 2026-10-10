import json
from hashlib import sha256
from uuid import UUID

import pytest

from organism import (
    CanonicalContinuationRegistry,
    ContinuationId,
    DuplicateCanonicalContinuation,
    InvalidContinuationCheckpoint,
    OrganismCheckpoint,
    OrganismContinuation,
    OrganismIdentity,
    OrganismTransitionState,
    StaleCanonicalContinuation,
    branch_continuation,
)


IDENTITY = OrganismIdentity.create(UUID("a0a0a0a0-0000-4000-8000-000000000001"))
CONTINUATION = ContinuationId.create(UUID("b0b0b0b0-0000-4000-8000-000000000001"))
BRANCH_IDENTITY = OrganismIdentity.create(UUID("a0a0a0a0-0000-4000-8000-000000000002"))
BRANCH_CONTINUATION = ContinuationId.create(UUID("b0b0b0b0-0000-4000-8000-000000000002"))


def continuation(identity=IDENTITY, continuation_id=CONTINUATION, step=0, state=b"private"):
    checkpoint = OrganismCheckpoint(identity, state)
    transition = OrganismTransitionState(checkpoint, organism_step=step)
    return OrganismContinuation(transition, continuation_id)


def test_continuation_checkpoint_round_trips_without_claiming_canonical_authority():
    saved = continuation()

    restored = OrganismContinuation.from_bytes(saved.to_bytes())

    assert restored == saved
    assert "is_canonical" not in json.loads(saved.to_bytes())


def test_registry_rejects_duplicate_live_restore_and_stale_checkpoint():
    initial = continuation()
    latest = continuation(step=1, state=b"next")
    registry = CanonicalContinuationRegistry()
    registry.register_genesis(initial)

    lease = registry.acquire(initial)
    with pytest.raises(DuplicateCanonicalContinuation, match="already live"):
        registry.acquire(initial)
    lease.commit(latest)
    lease.close()

    with pytest.raises(StaleCanonicalContinuation, match="not the current"):
        registry.acquire(initial)
    with registry.acquire(latest) as restored:
        assert restored.current == latest


def test_branch_requires_new_owner_state_and_records_ancestry():
    source = continuation(step=4)
    branch_state = OrganismTransitionState(
        OrganismCheckpoint(BRANCH_IDENTITY, b"owner-produced branch state"),
        organism_step=4,
    )

    branch = branch_continuation(
        source,
        new_identity=BRANCH_IDENTITY,
        new_continuation_id=BRANCH_CONTINUATION,
        branched_state=branch_state,
    )
    restored = OrganismContinuation.from_bytes(branch.to_bytes())

    assert restored.identity == BRANCH_IDENTITY
    assert restored.ancestry is not None
    assert restored.ancestry.parent_organism_id == source.identity
    assert restored.ancestry.parent_continuation_id == source.continuation_id
    assert restored.ancestry.parent_organism_step == 4
    assert restored.ancestry.source_checkpoint_sha256 == sha256(
        source.state.owner_checkpoint.to_bytes()
    ).hexdigest()

    registry = CanonicalContinuationRegistry()
    registry.register_genesis(continuation(step=0))
    with pytest.raises(ValueError, match="explicit ancestry"):
        registry.designate_branch_from(
            continuation(BRANCH_IDENTITY, BRANCH_CONTINUATION), source=source
        )
    wrong_source = continuation(
        identity=OrganismIdentity.create(UUID("a0a0a0a0-0000-4000-8000-000000000003")),
        step=4,
    )
    with pytest.raises(ValueError, match="parent identity"):
        registry.designate_branch_from(branch, source=wrong_source)
    registry.designate_branch_from(branch, source=source)
    with registry.acquire(branch) as lease:
        assert lease.current == branch


def test_registry_snapshot_restores_exact_owner_authority_state():
    initial = continuation()
    registry = CanonicalContinuationRegistry()
    registry.register_genesis(initial)

    restored = CanonicalContinuationRegistry.from_bytes(registry.to_bytes())

    with restored.acquire(initial) as lease:
        assert lease.current == initial


def test_registry_checkpoint_decoder_rejects_unknown_fields():
    saved = json.loads(CanonicalContinuationRegistry().to_bytes())
    saved["is_canonical"] = True

    with pytest.raises(InvalidContinuationCheckpoint, match="invalid shape"):
        CanonicalContinuationRegistry.from_bytes(json.dumps(saved).encode())
