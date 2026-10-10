"""Canonical Organism-owned longitudinal transition boundary.

This module owns Organism progression semantics. World may correlate an
accepted Organism transition with a World causal tick, but that correlation is
never the Organism progression counter. Embodiment references identify the
body context in which the Organism participates; somatic/body state remains
owned by Embodiment.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
from enum import StrEnum

from .checkpoint import InvalidOrganismCheckpoint, OrganismCheckpoint
from .identity import OrganismIdentity


class OrganismTransitionError(ValueError):
    """A proposed canonical Organism transition is invalid."""


class InvalidOrganismTransitionCheckpoint(ValueError):
    """A serialized canonical Organism transition state is invalid."""


class OrganismLifecycle(StrEnum):
    """Longitudinal Organism lifecycle, independent of body/presence lifecycle."""

    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    TERMINATED = "TERMINATED"


@dataclass(frozen=True, slots=True)
class LifecycleTransitionRecord:
    """Owner-recorded lifecycle transition and opaque request provenance."""

    transition_id: str
    from_state: OrganismLifecycle
    to_state: OrganismLifecycle
    request_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.transition_id, str) or not self.transition_id.strip():
            raise OrganismTransitionError("lifecycle transition_id must be non-empty")
        if not isinstance(self.request_ref, str) or not self.request_ref.strip():
            raise OrganismTransitionError("lifecycle request_ref must be non-empty")
        if not isinstance(self.from_state, OrganismLifecycle) or not isinstance(self.to_state, OrganismLifecycle):
            raise OrganismTransitionError("lifecycle transition states must be explicit")
        if (self.from_state, self.to_state) not in {
            (OrganismLifecycle.ACTIVE, OrganismLifecycle.DORMANT),
            (OrganismLifecycle.DORMANT, OrganismLifecycle.ACTIVE),
            (OrganismLifecycle.ACTIVE, OrganismLifecycle.TERMINATED),
            (OrganismLifecycle.DORMANT, OrganismLifecycle.TERMINATED),
        }:
            raise OrganismTransitionError("invalid or terminal Organism lifecycle transition")


@dataclass(frozen=True, slots=True)
class WorldCausalCorrelation:
    """Optional correlation to a World causal cut; never Organism time."""

    world_id: str
    causal_tick: int

    def __post_init__(self) -> None:
        if not isinstance(self.world_id, str) or not self.world_id.strip():
            raise OrganismTransitionError("world_id must be a non-empty string")
        if type(self.causal_tick) is not int or self.causal_tick < 0:
            raise OrganismTransitionError(
                "world causal tick must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class EmbodimentContextRef:
    """Identity-only reference to the current body context."""

    body_id: str
    embodiment_episode_id: str
    morphology_revision: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("body_id", self.body_id),
            ("embodiment_episode_id", self.embodiment_episode_id),
            ("morphology_revision", self.morphology_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise OrganismTransitionError(
                    f"{field_name} must be a non-empty string"
                )


@dataclass(frozen=True, slots=True)
class OrganismTransitionState:
    """Published state at the canonical Organism transition boundary.

    The owner checkpoint remains opaque outside Organism. The current bounded
    composition does not claim that every Organism subsystem is already
    included; completeness is governed separately by the owner-declared
    continuation closure contract.
    """

    owner_checkpoint: OrganismCheckpoint
    organism_step: int
    embodiment: EmbodimentContextRef | None = None
    world_correlation: WorldCausalCorrelation | None = None
    lifecycle: OrganismLifecycle | None = None
    lifecycle_history: tuple[LifecycleTransitionRecord, ...] = ()

    SCHEMA = "biopeia.organism-transition"
    SCHEMA_VERSION = 2

    def __post_init__(self) -> None:
        if not isinstance(self.owner_checkpoint, OrganismCheckpoint):
            raise OrganismTransitionError(
                "owner_checkpoint must be OrganismCheckpoint"
            )
        if type(self.organism_step) is not int or self.organism_step < 0:
            raise OrganismTransitionError(
                "organism_step must be a non-negative integer"
            )
        if self.embodiment is not None and not isinstance(
            self.embodiment, EmbodimentContextRef
        ):
            raise OrganismTransitionError(
                "embodiment must be EmbodimentContextRef or None"
            )
        if self.world_correlation is not None and not isinstance(
            self.world_correlation, WorldCausalCorrelation
        ):
            raise OrganismTransitionError(
                "world_correlation must be WorldCausalCorrelation or None"
            )
        if not isinstance(self.lifecycle, OrganismLifecycle):
            raise OrganismTransitionError("lifecycle must be explicitly supplied")
        if not isinstance(self.lifecycle_history, tuple) or any(
            not isinstance(record, LifecycleTransitionRecord) for record in self.lifecycle_history
        ):
            raise OrganismTransitionError("lifecycle_history must be a tuple of valid records")
        seen: set[str] = set()
        expected = OrganismLifecycle.ACTIVE
        for record in self.lifecycle_history:
            if record.transition_id in seen or record.from_state is not expected:
                raise OrganismTransitionError("lifecycle history is duplicated or discontinuous")
            seen.add(record.transition_id)
            expected = record.to_state
        if expected is not self.lifecycle:
            raise OrganismTransitionError("lifecycle does not match its recorded history")

    @property
    def identity(self) -> OrganismIdentity:
        return self.owner_checkpoint.identity

    def verify_dependency_closure(self, read_bytes) -> None:
        """Delegate verification of the owner-declared private closure."""
        self.owner_checkpoint.verify_dependency_closure(read_bytes)

    def to_bytes(self) -> bytes:
        """Encode the strict canonical transition-state envelope."""
        document = {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "identity": str(self.identity),
            "organism_step": self.organism_step,
            "owner_checkpoint_base64": base64.b64encode(
                self.owner_checkpoint.to_bytes()
            ).decode("ascii"),
            "embodiment": (
                None
                if self.embodiment is None
                else {
                    "body_id": self.embodiment.body_id,
                    "embodiment_episode_id": self.embodiment.embodiment_episode_id,
                    "morphology_revision": self.embodiment.morphology_revision,
                }
            ),
            "world_correlation": (
                None
                if self.world_correlation is None
                else {
                    "world_id": self.world_correlation.world_id,
                    "causal_tick": self.world_correlation.causal_tick,
                }
            ),
            "lifecycle": self.lifecycle.value,
            "lifecycle_history": [
                {
                    "transition_id": record.transition_id,
                    "from_state": record.from_state.value,
                    "to_state": record.to_state.value,
                    "request_ref": record.request_ref,
                }
                for record in self.lifecycle_history
            ],
        }
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OrganismTransitionState":
        """Restore a strict TARGET transition state; never default progression."""
        try:
            document = json.loads(payload)
            expected_keys = {
                "schema",
                "schema_version",
                "identity",
                "organism_step",
                "owner_checkpoint_base64",
                "embodiment",
                "world_correlation",
                "lifecycle",
                "lifecycle_history",
            }
            if not isinstance(document, dict) or set(document) != expected_keys:
                raise InvalidOrganismTransitionCheckpoint(
                    "transition checkpoint has an invalid shape"
                )
            schema_version = document["schema_version"]
            if (
                document["schema"] != cls.SCHEMA
                or type(schema_version) is not int
                or schema_version != cls.SCHEMA_VERSION
            ):
                raise InvalidOrganismTransitionCheckpoint(
                    "unsupported transition checkpoint schema or version"
                )
            organism_step = document["organism_step"]
            if type(organism_step) is not int or organism_step < 0:
                raise InvalidOrganismTransitionCheckpoint(
                    "organism_step must be a non-negative integer"
                )
            identity = OrganismIdentity.parse(document["identity"])
            checkpoint_payload = base64.b64decode(
                document["owner_checkpoint_base64"],
                validate=True,
            )
            owner_checkpoint = OrganismCheckpoint.from_bytes(checkpoint_payload)
            if owner_checkpoint.identity != identity:
                raise InvalidOrganismTransitionCheckpoint(
                    "transition and owner checkpoint identities differ"
                )

            raw_embodiment = document["embodiment"]
            if raw_embodiment is None:
                embodiment = None
            else:
                if not isinstance(raw_embodiment, dict) or set(raw_embodiment) != {
                    "body_id",
                    "embodiment_episode_id",
                    "morphology_revision",
                }:
                    raise InvalidOrganismTransitionCheckpoint(
                        "embodiment context has an invalid shape"
                    )
                embodiment = EmbodimentContextRef(**raw_embodiment)

            raw_world = document["world_correlation"]
            if raw_world is None:
                world_correlation = None
            else:
                if not isinstance(raw_world, dict) or set(raw_world) != {
                    "world_id",
                    "causal_tick",
                }:
                    raise InvalidOrganismTransitionCheckpoint(
                        "world correlation has an invalid shape"
                    )
                world_correlation = WorldCausalCorrelation(**raw_world)

            raw_history = document["lifecycle_history"]
            if not isinstance(raw_history, list):
                raise InvalidOrganismTransitionCheckpoint("lifecycle_history must be a list")
            history = []
            for raw_record in raw_history:
                if not isinstance(raw_record, dict) or set(raw_record) != {
                    "transition_id", "from_state", "to_state", "request_ref"
                }:
                    raise InvalidOrganismTransitionCheckpoint("lifecycle transition record has an invalid shape")
                history.append(LifecycleTransitionRecord(
                    transition_id=raw_record["transition_id"],
                    from_state=OrganismLifecycle(raw_record["from_state"]),
                    to_state=OrganismLifecycle(raw_record["to_state"]),
                    request_ref=raw_record["request_ref"],
                ))

            return cls(
                owner_checkpoint=owner_checkpoint,
                organism_step=organism_step,
                embodiment=embodiment,
                world_correlation=world_correlation,
                lifecycle=OrganismLifecycle(document["lifecycle"]),
                lifecycle_history=tuple(history),
            )
        except InvalidOrganismTransitionCheckpoint:
            raise
        except (
            TypeError,
            ValueError,
            KeyError,
            binascii.Error,
            json.JSONDecodeError,
            InvalidOrganismCheckpoint,
        ) as exc:
            raise InvalidOrganismTransitionCheckpoint(
                "transition checkpoint is malformed"
            ) from exc


@dataclass(frozen=True, slots=True)
class OrganismTransitionProposal:
    """Complete proposed next owner state for one accepted transition."""

    owner_checkpoint: OrganismCheckpoint
    embodiment: EmbodimentContextRef | None = None
    world_correlation: WorldCausalCorrelation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.owner_checkpoint, OrganismCheckpoint):
            raise OrganismTransitionError(
                "proposal checkpoint must be OrganismCheckpoint"
            )
        if self.embodiment is not None and not isinstance(
            self.embodiment, EmbodimentContextRef
        ):
            raise OrganismTransitionError(
                "proposal embodiment must be EmbodimentContextRef or None"
            )
        if self.world_correlation is not None and not isinstance(
            self.world_correlation, WorldCausalCorrelation
        ):
            raise OrganismTransitionError(
                "proposal world correlation must be WorldCausalCorrelation or None"
            )


def accept_organism_transition(
    current: OrganismTransitionState,
    proposal: OrganismTransitionProposal,
) -> OrganismTransitionState:
    """Validate and publish exactly one next longitudinal Organism state.

    The function is pure: validation failure leaves current unchanged. Callers
    that do not invoke this boundary for a World step do not advance
    organism_step.
    """
    if not isinstance(current, OrganismTransitionState):
        raise OrganismTransitionError("current must be OrganismTransitionState")
    if not isinstance(proposal, OrganismTransitionProposal):
        raise OrganismTransitionError(
            "proposal must be OrganismTransitionProposal"
        )
    if proposal.owner_checkpoint.identity != current.identity:
        raise OrganismTransitionError(
            "proposed state belongs to a different Organism identity"
        )
    if current.lifecycle is not OrganismLifecycle.ACTIVE:
        raise OrganismTransitionError("ordinary progression requires an ACTIVE Organism")

    return OrganismTransitionState(
        owner_checkpoint=proposal.owner_checkpoint,
        organism_step=current.organism_step + 1,
        embodiment=proposal.embodiment,
        world_correlation=proposal.world_correlation,
        lifecycle=current.lifecycle,
        lifecycle_history=current.lifecycle_history,
    )


def transition_organism_lifecycle(
    current: OrganismTransitionState,
    target: OrganismLifecycle,
    *,
    transition_id: str,
    request_ref: str,
    embodiment: EmbodimentContextRef | None = None,
) -> OrganismTransitionState:
    """Apply an explicit lifecycle request without advancing organism_step.

    Body death and World-presence termination are not inputs to this function;
    neither implies longitudinal Organism termination.
    """
    if not isinstance(current, OrganismTransitionState):
        raise OrganismTransitionError("current must be OrganismTransitionState")
    if not isinstance(target, OrganismLifecycle):
        raise OrganismTransitionError("target lifecycle must be explicit")
    if embodiment is not None and not isinstance(embodiment, EmbodimentContextRef):
        raise OrganismTransitionError("embodiment must be an EmbodimentContextRef when supplied")
    if embodiment is not None and not (
        current.lifecycle is OrganismLifecycle.DORMANT and target is OrganismLifecycle.ACTIVE
    ):
        raise OrganismTransitionError(
            "a new Embodiment context may be bound only when reactivating a DORMANT Organism"
        )
    record = LifecycleTransitionRecord(transition_id, current.lifecycle, target, request_ref)
    return OrganismTransitionState(
        owner_checkpoint=current.owner_checkpoint,
        organism_step=current.organism_step,
        embodiment=embodiment if embodiment is not None else current.embodiment,
        world_correlation=current.world_correlation,
        lifecycle=target,
        lifecycle_history=(*current.lifecycle_history, record),
    )
