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

from .checkpoint import InvalidOrganismCheckpoint, OrganismCheckpoint
from .identity import OrganismIdentity


class OrganismTransitionError(ValueError):
    """A proposed canonical Organism transition is invalid."""


class InvalidOrganismTransitionCheckpoint(ValueError):
    """A serialized canonical Organism transition state is invalid."""


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

    SCHEMA = "biopeia.organism-transition"
    SCHEMA_VERSION = 1

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

            return cls(
                owner_checkpoint=owner_checkpoint,
                organism_step=organism_step,
                embodiment=embodiment,
                world_correlation=world_correlation,
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

    return OrganismTransitionState(
        owner_checkpoint=proposal.owner_checkpoint,
        organism_step=current.organism_step + 1,
        embodiment=proposal.embodiment,
        world_correlation=proposal.world_correlation,
    )
