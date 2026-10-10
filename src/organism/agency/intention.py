"""ActionIntent: the persistent executive commitment to a consequence (§39-§44).

KNOW    "I can do C"                (MotorCompetence)
CONSIDER "C is relevant/possible"   (ActionAffordance)
INTEND  "I will attempt C now"      (ActionIntent)
ACT     "C is being executed"       (ActionCommitment -> controller -> MotorCommand)

An intent answers WHAT is being attempted, never HOW: it holds a competence
reference and an anticipated organism-owned effect, never actuator ids,
trajectories, gains or a scalar reward.  It has no motor authority; only the
ActionArbitrator grants that, through an ActionCommitment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class IntentStatus(StrEnum):
    PENDING = "pending"

    ACTIVE = "active"

    SATISFIED = "satisfied"
    FAILED = "failed"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    INVALIDATED = "invalidated"


TERMINAL_INTENT_STATUSES = frozenset(
    {
        IntentStatus.SATISFIED,
        IntentStatus.FAILED,
        IntentStatus.REJECTED,
        IntentStatus.INTERRUPTED,
        IntentStatus.INVALIDATED,
    }
)

# §42: the only legal lifecycle edges.
_TRANSITIONS: dict[IntentStatus, frozenset[IntentStatus]] = {
    IntentStatus.PENDING: frozenset(
        {IntentStatus.ACTIVE, IntentStatus.REJECTED, IntentStatus.INVALIDATED}
    ),
    IntentStatus.ACTIVE: frozenset(
        {
            IntentStatus.SATISFIED,
            IntentStatus.FAILED,
            IntentStatus.INTERRUPTED,
            IntentStatus.INVALIDATED,
        }
    ),
}


class AdmissionRoute(StrEnum):
    """How a represented possibility was selected for realization (§48, §73-§74)."""

    COGNITIVE = "cognitive"  # primitive readout + affordance + executive admission
    PROSPECTIVE = "prospective"  # model-based ProspectiveAgency deliberation
    GENERATIVE = "generative"  # anticipated effect from Generative Cognition


_FORBIDDEN_REF_PREFIXES = ("actuator.", "motor.", "channel.", "joint.", "controller.")


def _unit(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


@dataclass(slots=True)
class ActionIntent:
    intent_id: str

    competence_id: str
    anticipated_effect_id: str | None

    context_ref: str | None
    embodiment_id: str | None

    origin_refs: tuple[str, ...]
    prediction_ref: str | None

    confidence: float
    epistemic_relevance: float
    homeostatic_relevance: float

    supporting_affordance_id: str | None

    created_tick: int
    activated_tick: int | None
    last_progress_tick: int

    status: IntentStatus
    termination_reason: str | None

    admission: AdmissionRoute = AdmissionRoute.COGNITIVE

    def __post_init__(self) -> None:
        if not self.intent_id.startswith("intent."):
            raise ValueError("intent_id must be opaque")
        if not self.competence_id or self.competence_id.startswith(_FORBIDDEN_REF_PREFIXES):
            raise ValueError("an intent references an acquired competence, never an actuator")
        if self.anticipated_effect_id is not None and not self.anticipated_effect_id.startswith(
            "effect."
        ):
            raise ValueError("anticipated effect must be organism-owned")
        if any(ref.startswith(_FORBIDDEN_REF_PREFIXES) for ref in self.origin_refs):
            raise ValueError("intent origins never carry actuator or controller semantics")
        for name in ("confidence", "epistemic_relevance", "homeostatic_relevance"):
            setattr(self, name, _unit(getattr(self, name), name))
        if self.created_tick < 0 or self.last_progress_tick < self.created_tick:
            raise ValueError("invalid intent ticks")
        if self.activated_tick is not None and self.activated_tick < self.created_tick:
            raise ValueError("an intent cannot activate before it is formed")
        self.status = IntentStatus(self.status)
        self.admission = AdmissionRoute(self.admission)
        self._validate_lifecycle_state()

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_INTENT_STATUSES

    def _validate_lifecycle_state(self) -> None:
        if self.status is IntentStatus.PENDING:
            if self.activated_tick is not None or self.termination_reason is not None:
                raise ValueError("pending intent cannot have activation or termination state")
        elif self.status is IntentStatus.ACTIVE:
            if self.activated_tick is None or self.termination_reason is not None:
                raise ValueError("active intent requires activation and no termination reason")
        elif self.terminal:
            if not self.termination_reason:
                raise ValueError("terminal intent requires a termination reason")
            if self.status in {
                IntentStatus.SATISFIED,
                IntentStatus.FAILED,
                IntentStatus.INTERRUPTED,
            } and self.activated_tick is None:
                raise ValueError(f"{self.status.value} intent requires prior activation")
            if self.status is IntentStatus.REJECTED and self.activated_tick is not None:
                raise ValueError("rejected intent cannot have been activated")
        else:  # Defensive if a future enum member is added without lifecycle rules.
            raise ValueError("intent status has no lifecycle contract")

        if self.activated_tick is not None and self.last_progress_tick < self.activated_tick:
            raise ValueError("intent progress cannot precede activation")

    def transition(self, status: IntentStatus, *, tick: int, reason: str | None = None) -> None:
        if status not in _TRANSITIONS.get(self.status, frozenset()):
            raise ValueError(f"illegal intent transition {self.status} -> {status}")
        if type(tick) is not int:
            raise TypeError("intent transition tick must be an integer")
        if tick < self.last_progress_tick:
            raise ValueError("intent transition tick cannot precede prior progress")
        if status is IntentStatus.ACTIVE:
            self.activated_tick = tick
            self.last_progress_tick = tick
        else:
            if not reason:
                raise ValueError("a terminal intent requires a termination reason")
            self.termination_reason = reason
        self.status = status

    def checkpoint(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "competence_id": self.competence_id,
            "anticipated_effect_id": self.anticipated_effect_id,
            "context_ref": self.context_ref,
            "embodiment_id": self.embodiment_id,
            "origin_refs": list(self.origin_refs),
            "prediction_ref": self.prediction_ref,
            "confidence": self.confidence,
            "epistemic_relevance": self.epistemic_relevance,
            "homeostatic_relevance": self.homeostatic_relevance,
            "supporting_affordance_id": self.supporting_affordance_id,
            "created_tick": self.created_tick,
            "activated_tick": self.activated_tick,
            "last_progress_tick": self.last_progress_tick,
            "status": self.status.value,
            "termination_reason": self.termination_reason,
            "admission": self.admission.value,
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "ActionIntent":
        if not isinstance(payload, Mapping):
            raise TypeError("action intent checkpoint must be a mapping")

        expected = {
            "intent_id",
            "competence_id",
            "anticipated_effect_id",
            "context_ref",
            "embodiment_id",
            "origin_refs",
            "prediction_ref",
            "confidence",
            "epistemic_relevance",
            "homeostatic_relevance",
            "supporting_affordance_id",
            "created_tick",
            "activated_tick",
            "last_progress_tick",
            "status",
            "termination_reason",
            "admission",
        }
        if any(type(key) is not str for key in payload):
            raise TypeError("action intent checkpoint field names must be strings")
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"action intent checkpoint has missing or unknown fields: "
                f"missing={missing}, unknown={unknown}"
            )

        def required_str(key: str) -> str:
            value = payload[key]
            if type(value) is not str:
                raise TypeError(f"{key} must be a string")
            return value

        def nullable_str(key: str) -> str | None:
            value = payload[key]
            if value is not None and type(value) is not str:
                raise TypeError(f"{key} must be a string or null")
            return value

        def unit_float(key: str) -> float:
            value = payload[key]
            if type(value) is not float:
                raise TypeError(f"{key} must be a float")
            return value

        def required_tick(key: str) -> int:
            value = payload[key]
            if type(value) is not int:
                raise TypeError(f"{key} must be an integer")
            return value

        origin_refs = payload["origin_refs"]
        if type(origin_refs) is not list or any(type(ref) is not str for ref in origin_refs):
            raise TypeError("origin_refs must be a list of strings")

        activated = payload["activated_tick"]
        if activated is not None and type(activated) is not int:
            raise TypeError("activated_tick must be an integer or null")

        status = required_str("status")
        admission = required_str("admission")
        return cls(
            intent_id=required_str("intent_id"),
            competence_id=required_str("competence_id"),
            anticipated_effect_id=nullable_str("anticipated_effect_id"),
            context_ref=nullable_str("context_ref"),
            embodiment_id=nullable_str("embodiment_id"),
            origin_refs=tuple(origin_refs),
            prediction_ref=nullable_str("prediction_ref"),
            confidence=unit_float("confidence"),
            epistemic_relevance=unit_float("epistemic_relevance"),
            homeostatic_relevance=unit_float("homeostatic_relevance"),
            supporting_affordance_id=nullable_str("supporting_affordance_id"),
            created_tick=required_tick("created_tick"),
            activated_tick=activated,
            last_progress_tick=required_tick("last_progress_tick"),
            status=IntentStatus(status),
            termination_reason=nullable_str("termination_reason"),
            admission=AdmissionRoute(admission),
        )


__all__ = [
    "ActionIntent",
    "AdmissionRoute",
    "IntentStatus",
    "TERMINAL_INTENT_STATUSES",
]
