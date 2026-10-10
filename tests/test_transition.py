import json
from hashlib import sha256
from uuid import UUID

import pytest

from organism import (
    CheckpointDependency,
    EmbodimentContextRef,
    InvalidOrganismTransitionCheckpoint,
    OrganismCheckpoint,
    OrganismIdentity,
    OrganismLifecycle,
    OrganismTransitionError,
    OrganismTransitionProposal,
    OrganismTransitionState,
    WorldCausalCorrelation,
    accept_organism_transition,
    transition_organism_lifecycle,
)


IDENTITY = OrganismIdentity.create(
    UUID("2f750988-3437-41b4-bf83-f70f876d4807")
)


def owner_checkpoint(
    *,
    identity: OrganismIdentity = IDENTITY,
    private_state: bytes = b"owner-private-state-v1",
) -> OrganismCheckpoint:
    dependency = CheckpointDependency(
        "sha256/private-model",
        sha256(b"private-model").hexdigest(),
    )
    return OrganismCheckpoint(identity, private_state, (dependency,))


def state() -> OrganismTransitionState:
    return OrganismTransitionState(
        owner_checkpoint=owner_checkpoint(),
        organism_step=7,
        embodiment=EmbodimentContextRef(
            "body-a",
            "episode-a",
            "morphology-r1",
        ),
        world_correlation=WorldCausalCorrelation("world-a", 19),
        lifecycle=OrganismLifecycle.ACTIVE,
    )


def test_accepted_transition_advances_organism_step_exactly_once() -> None:
    current = state()
    proposed = OrganismTransitionProposal(
        owner_checkpoint=owner_checkpoint(private_state=b"next-private-state"),
        embodiment=current.embodiment,
        world_correlation=WorldCausalCorrelation("world-a", 20),
    )

    accepted = accept_organism_transition(current, proposed)

    assert current.organism_step == 7
    assert accepted.organism_step == 8
    assert accepted.identity == current.identity
    assert accepted.world_correlation == WorldCausalCorrelation("world-a", 20)


def test_foreign_proposal_is_rejected_without_advancing_current_state() -> None:
    current = state()
    foreign = OrganismIdentity.create(
        UUID("460423ef-7b4c-4617-94b5-6cf8a4f687a7")
    )
    proposal = OrganismTransitionProposal(
        owner_checkpoint=owner_checkpoint(identity=foreign),
        embodiment=current.embodiment,
    )

    with pytest.raises(OrganismTransitionError, match="different Organism"):
        accept_organism_transition(current, proposal)

    assert current.organism_step == 7
    assert current.identity == IDENTITY


def test_world_step_without_organism_transition_does_not_advance() -> None:
    current = state()
    after_world_only_activity = current

    assert after_world_only_activity == current
    assert after_world_only_activity.organism_step == 7


def test_world_causal_tick_is_correlation_not_organism_step() -> None:
    current = state()
    accepted = accept_organism_transition(
        current,
        OrganismTransitionProposal(
            owner_checkpoint=owner_checkpoint(private_state=b"next"),
            embodiment=current.embodiment,
            world_correlation=WorldCausalCorrelation("world-a", 200),
        ),
    )

    assert accepted.organism_step == 8
    assert accepted.world_correlation is not None
    assert accepted.world_correlation.causal_tick == 200


def test_reembodiment_preserves_identity_and_longitudinal_state() -> None:
    current = state()
    replacement_body = EmbodimentContextRef(
        "body-b",
        "episode-b",
        "morphology-r2",
    )

    dormant = transition_organism_lifecycle(
        current,
        OrganismLifecycle.DORMANT,
        transition_id="lifecycle-1",
        request_ref="body-a-terminated",
    )
    accepted = transition_organism_lifecycle(
        dormant,
        OrganismLifecycle.ACTIVE,
        transition_id="lifecycle-2",
        request_ref="replacement-body-accepted",
        embodiment=replacement_body,
    )

    assert accepted.identity == current.identity
    assert accepted.owner_checkpoint == current.owner_checkpoint
    assert accepted.organism_step == current.organism_step
    assert accepted.embodiment == replacement_body
    assert current.embodiment != replacement_body


def test_ordinary_transition_cannot_replace_embodiment_context() -> None:
    current = state()
    replacement_body = EmbodimentContextRef(
        "body-b",
        "episode-b",
        "morphology-r2",
    )

    with pytest.raises(OrganismTransitionError, match="lifecycle transition"):
        accept_organism_transition(
            current,
            OrganismTransitionProposal(
                owner_checkpoint=current.owner_checkpoint,
                embodiment=replacement_body,
                world_correlation=WorldCausalCorrelation("world-a", 20),
            ),
        )

    assert current.organism_step == 7
    assert current.embodiment == EmbodimentContextRef(
        "body-a", "episode-a", "morphology-r1"
    )


def test_transition_checkpoint_round_trips_progression_and_context() -> None:
    saved = state()

    restored = OrganismTransitionState.from_bytes(saved.to_bytes())

    assert restored == saved


@pytest.mark.parametrize("bad_value", [None, True, -1, 1.5, "7"])
def test_transition_checkpoint_rejects_missing_or_invalid_organism_step(
    bad_value,
) -> None:
    document = json.loads(state().to_bytes())
    if bad_value is None:
        del document["organism_step"]
    else:
        document["organism_step"] = bad_value

    with pytest.raises(InvalidOrganismTransitionCheckpoint):
        OrganismTransitionState.from_bytes(json.dumps(document).encode())


def test_transition_checkpoint_rejects_foreign_nested_identity() -> None:
    document = json.loads(state().to_bytes())
    document["identity"] = "460423ef-7b4c-4617-94b5-6cf8a4f687a7"

    with pytest.raises(
        InvalidOrganismTransitionCheckpoint,
        match="identities differ",
    ):
        OrganismTransitionState.from_bytes(json.dumps(document).encode())


def test_dependency_verification_remains_owner_declared() -> None:
    saved = state()

    saved.verify_dependency_closure(
        lambda locator: (
            b"private-model"
            if locator == "sha256/private-model"
            else (_ for _ in ()).throw(KeyError(locator))
        )
    )


def test_body_death_or_presence_loss_does_not_implicitly_terminate_organism() -> None:
    current = state()
    dormant = transition_organism_lifecycle(
        current,
        OrganismLifecycle.DORMANT,
        transition_id="lifecycle-1",
        request_ref="owner-request-1",
    )

    assert dormant.lifecycle is OrganismLifecycle.DORMANT
    assert dormant.organism_step == current.organism_step
    assert dormant.identity == current.identity
    assert dormant.lifecycle_history[-1].request_ref == "owner-request-1"


def test_dormant_reembodiment_can_reactivate_same_longitudinal_organism() -> None:
    dormant = transition_organism_lifecycle(
        state(), OrganismLifecycle.DORMANT,
        transition_id="lifecycle-1", request_ref="owner-request-1",
    )
    active = transition_organism_lifecycle(
        dormant, OrganismLifecycle.ACTIVE,
        transition_id="lifecycle-2", request_ref="reembodiment-request-2",
    )

    assert active.identity == dormant.identity
    assert active.owner_checkpoint == dormant.owner_checkpoint
    assert active.organism_step == dormant.organism_step
    assert [record.to_state for record in active.lifecycle_history] == [
        OrganismLifecycle.DORMANT, OrganismLifecycle.ACTIVE
    ]


def test_reactivation_can_bind_new_explicit_embodiment_context() -> None:
    old = EmbodimentContextRef("body-a", "episode-a", "morph-1")
    new = EmbodimentContextRef("body-b", "episode-b", "morph-2")
    current = OrganismTransitionState(
        owner_checkpoint=state().owner_checkpoint,
        organism_step=7,
        embodiment=old,
        world_correlation=state().world_correlation,
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    dormant = transition_organism_lifecycle(
        current, OrganismLifecycle.DORMANT,
        transition_id="lifecycle-1", request_ref="owner-request-1",
    )
    active = transition_organism_lifecycle(
        dormant, OrganismLifecycle.ACTIVE,
        transition_id="lifecycle-2", request_ref="reembodiment-request-2",
        embodiment=new,
    )

    assert active.identity == current.identity
    assert active.organism_step == current.organism_step
    assert active.owner_checkpoint == current.owner_checkpoint
    assert active.embodiment == new


@pytest.mark.parametrize("target", [OrganismLifecycle.ACTIVE, OrganismLifecycle.DORMANT])
def test_terminated_organism_cannot_transition_again(target) -> None:
    terminated = transition_organism_lifecycle(
        state(), OrganismLifecycle.TERMINATED,
        transition_id="lifecycle-1", request_ref="owner-request-1",
    )
    with pytest.raises(OrganismTransitionError, match="invalid or terminal"):
        transition_organism_lifecycle(
            terminated, target, transition_id="lifecycle-2", request_ref="owner-request-2"
        )


@pytest.mark.parametrize("lifecycle", [OrganismLifecycle.DORMANT, OrganismLifecycle.TERMINATED])
def test_ordinary_progression_rejected_outside_active(lifecycle) -> None:
    current = state()
    if lifecycle is OrganismLifecycle.DORMANT:
        current = transition_organism_lifecycle(
            current, lifecycle, transition_id="lifecycle-1", request_ref="owner-request-1"
        )
    else:
        current = transition_organism_lifecycle(
            current, lifecycle, transition_id="lifecycle-1", request_ref="owner-request-1"
        )
    with pytest.raises(OrganismTransitionError, match="requires an ACTIVE"):
        accept_organism_transition(
            current,
            OrganismTransitionProposal(owner_checkpoint=current.owner_checkpoint),
        )


def test_restore_rejects_checkpoint_without_valid_lifecycle() -> None:
    document = json.loads(state().to_bytes())
    del document["lifecycle"]
    with pytest.raises(InvalidOrganismTransitionCheckpoint):
        OrganismTransitionState.from_bytes(json.dumps(document).encode())
