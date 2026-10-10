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
    OrganismTransitionError,
    OrganismTransitionProposal,
    OrganismTransitionState,
    WorldCausalCorrelation,
    accept_organism_transition,
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

    accepted = accept_organism_transition(
        current,
        OrganismTransitionProposal(
            owner_checkpoint=current.owner_checkpoint,
            embodiment=replacement_body,
            world_correlation=WorldCausalCorrelation("world-a", 20),
        ),
    )

    assert accepted.identity == current.identity
    assert accepted.owner_checkpoint == current.owner_checkpoint
    assert accepted.organism_step == current.organism_step + 1
    assert accepted.embodiment == replacement_body
    assert current.embodiment != replacement_body


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
