"""Temporal ownership of sensorimotor action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CommitmentStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    INCOMPATIBLE = "incompatible"
    INVALIDATED = "invalidated"


@dataclass(slots=True)
class ActionCommitment:
    SCHEMA_VERSION = 1
    commitment_id: str
    proposal_id: str
    effect_target_id: str | None
    competence_id: str | None
    started_tick: int
    controller_id: str
    surface_fingerprint: str
    embodiment_id: str | None = None
    intent_id: str | None = None
    interruptibility: float = 1.0
    minimum_duration: int = 0
    maximum_duration: int | None = None
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    ended_tick: int | None = None
    end_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.commitment_id or not self.proposal_id or not self.controller_id:
            raise ValueError("commitment identifiers must not be empty")
        if not self.surface_fingerprint:
            raise ValueError("commitment requires a surface fingerprint")
        if self.started_tick < 0 or self.minimum_duration < 0:
            raise ValueError("commitment ticks must be non-negative")
        if self.maximum_duration is not None and self.maximum_duration < self.minimum_duration:
            raise ValueError("maximum_duration must be >= minimum_duration")
        if not 0.0 <= float(self.interruptibility) <= 1.0:
            raise ValueError("interruptibility must be within [0, 1]")

    @property
    def active(self) -> bool:
        return self.status is CommitmentStatus.ACTIVE

    def compatible_with(
        self,
        fingerprint: str | None,
        *,
        embodiment_id: str | None = None,
    ) -> bool:
        if fingerprint is None or fingerprint != self.surface_fingerprint:
            return False
        if (
            self.embodiment_id is not None
            and embodiment_id is not None
            and self.embodiment_id != embodiment_id
        ):
            return False
        return True

    def terminate(self, *, tick: int, status: CommitmentStatus, reason: str) -> None:
        if status is CommitmentStatus.ACTIVE:
            raise ValueError("terminate requires a terminal status")
        if not self.active:
            return
        if tick < self.started_tick:
            raise ValueError("end tick precedes commitment start")
        self.status = status
        self.ended_tick = tick
        self.end_reason = reason

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "commitment_id": self.commitment_id,
            "proposal_id": self.proposal_id,
            "effect_target_id": self.effect_target_id,
            "competence_id": self.competence_id,
            "started_tick": self.started_tick,
            "controller_id": self.controller_id,
            "surface_fingerprint": self.surface_fingerprint,
            "embodiment_id": self.embodiment_id,
            "intent_id": self.intent_id,
            "interruptibility": self.interruptibility,
            "minimum_duration": self.minimum_duration,
            "maximum_duration": self.maximum_duration,
            "status": self.status.value,
            "ended_tick": self.ended_tick,
            "end_reason": self.end_reason,
        }

    @classmethod
    def restore(
        cls,
        payload: dict[str, Any],
    ) -> "ActionCommitment":
        required = {
            "schema_version",
            "commitment_id",
            "proposal_id",
            "effect_target_id",
            "competence_id",
            "started_tick",
            "controller_id",
            "surface_fingerprint",
            "embodiment_id",
            "intent_id",
            "interruptibility",
            "minimum_duration",
            "maximum_duration",
            "status",
            "ended_tick",
            "end_reason",
        }
        if set(payload) != required:
            raise ValueError("action commitment checkpoint fields do not match the current schema")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported action commitment checkpoint schema")
        for field in ("commitment_id", "proposal_id", "controller_id", "surface_fingerprint"):
            value = payload[field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"action commitment {field} must be a non-empty string")
        for field in (
            "effect_target_id",
            "competence_id",
            "embodiment_id",
            "intent_id",
            "end_reason",
        ):
            if payload[field] is not None and not isinstance(payload[field], str):
                raise ValueError(f"action commitment {field} must be a string or null")
        for field in ("started_tick", "minimum_duration"):
            if isinstance(payload[field], bool) or not isinstance(payload[field], int):
                raise ValueError(f"action commitment {field} must be an integer")
        if payload["maximum_duration"] is not None and (
            isinstance(payload["maximum_duration"], bool)
            or not isinstance(payload["maximum_duration"], int)
        ):
            raise ValueError("action commitment maximum_duration must be an integer or null")
        if payload["ended_tick"] is not None and (
            isinstance(payload["ended_tick"], bool) or not isinstance(payload["ended_tick"], int)
        ):
            raise ValueError("action commitment ended_tick must be an integer or null")
        if isinstance(payload["interruptibility"], bool) or not isinstance(
            payload["interruptibility"], (int, float)
        ):
            raise ValueError("action commitment interruptibility must be numeric")
        obj = cls(
            commitment_id=payload["commitment_id"],
            proposal_id=payload["proposal_id"],
            effect_target_id=payload["effect_target_id"],
            competence_id=payload["competence_id"],
            started_tick=payload["started_tick"],
            controller_id=payload["controller_id"],
            surface_fingerprint=payload["surface_fingerprint"],
            embodiment_id=payload["embodiment_id"],
            intent_id=payload["intent_id"],
            interruptibility=float(payload["interruptibility"]),
            minimum_duration=payload["minimum_duration"],
            maximum_duration=payload["maximum_duration"],
            status=CommitmentStatus(payload["status"]),
        )
        obj.ended_tick = payload["ended_tick"]
        obj.end_reason = payload["end_reason"]
        return obj
