import json
from hashlib import sha256
from uuid import UUID

import pytest

from organism import (
    CanonicalContinuationRegistry,
    ContinuationError,
    ContinuationId,
    DuplicateCanonicalContinuation,
    InvalidContinuationCheckpoint,
    OrganismCheckpoint,
    OrganismContinuation,
    OrganismIdentity,
    OrganismLifecycle,
    EmbodimentContextRef,
    OrganismTransitionState,
    StaleCanonicalContinuation,
    OrganismTransitionProposal,
    accept_organism_transition,
    branch_continuation,
    transition_organism_lifecycle,
)


IDENTITY = OrganismIdentity.create(UUID("a0a0a0a0-0000-4000-8000-000000000001"))
CONTINUATION = ContinuationId.create(UUID("b0b0b0b0-0000-4000-8000-000000000001"))
BRANCH_IDENTITY = OrganismIdentity.create(UUID("a0a0a0a0-0000-4000-8000-000000000002"))
BRANCH_CONTINUATION = ContinuationId.create(UUID("b0b0b0b0-0000-4000-8000-000000000002"))


def continuation(identity=IDENTITY, continuation_id=CONTINUATION, step=0, state=b"private"):
    checkpoint = OrganismCheckpoint(identity, state)
    transition = OrganismTransitionState(checkpoint, organism_step=step, lifecycle=OrganismLifecycle.ACTIVE)
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
        lifecycle=OrganismLifecycle.ACTIVE,
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


def test_registry_commits_lifecycle_without_advancing_organism_step():
    initial = continuation()
    registry = CanonicalContinuationRegistry()
    registry.register_genesis(initial)

    with registry.acquire(initial) as lease:
        dormant_state = transition_organism_lifecycle(
            lease.current.state,
            OrganismLifecycle.DORMANT,
            transition_id="lifecycle-1",
            request_ref="owner-request-1",
        )
        dormant = OrganismContinuation(dormant_state, lease.current.continuation_id)
        lease.commit(dormant)
        assert lease.current.organism_step == initial.organism_step
        assert lease.current.state.lifecycle is OrganismLifecycle.DORMANT

    with registry.acquire(dormant) as restored:
        assert restored.current == dormant


@pytest.mark.parametrize("lifecycle", [OrganismLifecycle.DORMANT, OrganismLifecycle.TERMINATED])
def test_registry_rejects_ordinary_step_committed_directly_from_non_active_state(lifecycle):
    initial = continuation()
    registry = CanonicalContinuationRegistry()
    registry.register_genesis(initial)
    with registry.acquire(initial) as lease:
        non_active_state = transition_organism_lifecycle(
            lease.current.state,
            lifecycle,
            transition_id="lifecycle-1",
            request_ref="owner-request-1",
        )
        non_active = OrganismContinuation(non_active_state, lease.current.continuation_id)
        lease.commit(non_active)

        forged_progression_state = OrganismTransitionState(
            owner_checkpoint=non_active.state.owner_checkpoint,
            organism_step=non_active.organism_step + 1,
            lifecycle=lifecycle,
            lifecycle_history=non_active.state.lifecycle_history,
        )
        forged_progression = OrganismContinuation(
            forged_progression_state, non_active.continuation_id
        )
        with pytest.raises(ContinuationError, match="one active step"):
            lease.commit(forged_progression)


def test_registry_rejects_direct_ordinary_commit_with_changed_embodiment_context():
    original_context = EmbodimentContextRef("body-a", "episode-a", "morph-1")
    changed_context = EmbodimentContextRef("body-b", "episode-b", "morph-2")
    initial_state = OrganismTransitionState(
        OrganismCheckpoint(IDENTITY, b"private"),
        organism_step=0,
        embodiment=original_context,
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    initial = OrganismContinuation(initial_state, CONTINUATION)
    registry = CanonicalContinuationRegistry()
    registry.register_genesis(initial)

    with registry.acquire(initial) as lease:
        forged_state = OrganismTransitionState(
            initial.state.owner_checkpoint,
            organism_step=initial.organism_step + 1,
            embodiment=changed_context,
            world_correlation=initial.state.world_correlation,
            lifecycle=OrganismLifecycle.ACTIVE,
            lifecycle_history=initial.state.lifecycle_history,
        )
        forged = OrganismContinuation(forged_state, CONTINUATION)

        with pytest.raises(ContinuationError, match="one active step"):
            lease.commit(forged)

        assert lease.current == initial

    with registry.acquire(initial) as lease:
        assert lease.current == initial


def test_registry_allows_reactivation_to_bind_new_embodiment_context():
    original_context = EmbodimentContextRef("body-a", "episode-a", "morph-1")
    checkpoint = OrganismCheckpoint(IDENTITY, b"private")
    initial_state = OrganismTransitionState(
        checkpoint, organism_step=0, embodiment=original_context,
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    initial = OrganismContinuation(initial_state, CONTINUATION)
    registry = CanonicalContinuationRegistry()
    registry.register_genesis(initial)
    new_context = EmbodimentContextRef("body-b", "episode-b", "morph-2")

    with registry.acquire(initial) as lease:
        dormant_state = transition_organism_lifecycle(
            lease.current.state, OrganismLifecycle.DORMANT,
            transition_id="lifecycle-1", request_ref="owner-request-1",
        )
        dormant = OrganismContinuation(dormant_state, CONTINUATION)
        lease.commit(dormant)
        active_state = transition_organism_lifecycle(
            dormant_state, OrganismLifecycle.ACTIVE,
            transition_id="lifecycle-2", request_ref="reembodiment-request-2",
            embodiment=new_context,
        )
        active = OrganismContinuation(active_state, CONTINUATION)
        lease.commit(active)

    assert active.identity == initial.identity
    assert active.organism_step == initial.organism_step
    assert active.state.owner_checkpoint == initial.state.owner_checkpoint
    assert active.state.embodiment == new_context


def test_reembodiment_context_and_longitudinal_state_survive_continuation_restore():
    original_context = EmbodimentContextRef("body-a", "episode-a", "morph-1")
    new_context = EmbodimentContextRef("body-b", "episode-b", "morph-2")
    checkpoint = OrganismCheckpoint(IDENTITY, b"longitudinal-private-state")
    initial_state = OrganismTransitionState(
        checkpoint,
        organism_step=0,
        embodiment=original_context,
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    initial = OrganismContinuation(initial_state, CONTINUATION)
    registry = CanonicalContinuationRegistry()
    registry.register_genesis(initial)

    with registry.acquire(initial) as lease:
        progressed_state = accept_organism_transition(
            lease.current.state,
            OrganismTransitionProposal(
                owner_checkpoint=lease.current.state.owner_checkpoint,
                embodiment=original_context,
            ),
        )
        progressed = OrganismContinuation(progressed_state, CONTINUATION)
        lease.commit(progressed)

    with registry.acquire(progressed) as lease:
        dormant_state = transition_organism_lifecycle(
            lease.current.state,
            OrganismLifecycle.DORMANT,
            transition_id="lifecycle-1",
            request_ref="prior-presence-ended",
        )
        dormant = OrganismContinuation(dormant_state, CONTINUATION)
        lease.commit(dormant)

    restored_registry = CanonicalContinuationRegistry.from_bytes(registry.to_bytes())
    with restored_registry.acquire(dormant) as lease:
        active_state = transition_organism_lifecycle(
            lease.current.state,
            OrganismLifecycle.ACTIVE,
            transition_id="lifecycle-2",
            request_ref="explicit-reembodiment",
            embodiment=new_context,
        )
        active = OrganismContinuation(active_state, CONTINUATION)
        lease.commit(active)

    restored_active = OrganismContinuation.from_bytes(active.to_bytes())
    assert restored_active.identity == initial.identity
    assert restored_active.continuation_id == initial.continuation_id
    assert restored_active.organism_step == progressed.organism_step
    assert restored_active.state.owner_checkpoint == initial.state.owner_checkpoint
    assert restored_active.state.embodiment == new_context
    assert restored_active.state.lifecycle is OrganismLifecycle.ACTIVE


def test_registry_checkpoint_decoder_rejects_unknown_fields():
    saved = json.loads(CanonicalContinuationRegistry().to_bytes())
    saved["is_canonical"] = True

    with pytest.raises(InvalidContinuationCheckpoint, match="invalid shape"):
        CanonicalContinuationRegistry.from_bytes(json.dumps(saved).encode())
