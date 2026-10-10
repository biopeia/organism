from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True, slots=True)
class ActuatorEvidenceContext:
    """Embodiment identity that bounds current actuator evidence applicability.

    This is a reference to owner-supplied identity, not an Organism-owned body
    model. The fields intentionally mirror the explicit Embodiment contract.
    """

    body_id: str
    embodiment_episode_id: str
    morphology_revision: str
    surface_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "body_id",
            "embodiment_episode_id",
            "morphology_revision",
            "surface_fingerprint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def to_payload(self) -> dict[str, str]:
        return {
            "body_id": self.body_id,
            "embodiment_episode_id": self.embodiment_episode_id,
            "morphology_revision": self.morphology_revision,
            "surface_fingerprint": self.surface_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ActuatorEvidenceContext:
        expected = {
            "body_id",
            "embodiment_episode_id",
            "morphology_revision",
            "surface_fingerprint",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("ActuatorEvidenceContext payload fields do not match the current schema")
        return cls(**payload)


__all__ = ["ActuatorEvidenceContext"]
