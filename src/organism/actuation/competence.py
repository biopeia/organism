"""Evidence-derived motor competence abstractions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum


class CompetenceMaturity(StrEnum):
    CANDIDATE = "candidate"
    EMERGING = "emerging"
    ESTABLISHED = "established"
    ROBUST = "robust"


@dataclass(frozen=True, slots=True)
class CompetenceGate:
    """Evidence an organism requires before a competence counts as established.

    Organism configuration, carried by the canonical organism profile. A
    runtime checkpoint may record the effective configuration for exact
    continuation; it is not competence evidence and does not transfer
    configuration authority to the checkpoint.
    """

    min_support: int = 2
    min_consistency: float = 0.60

    def __post_init__(self) -> None:
        if isinstance(self.min_support, bool) or not 2 <= int(self.min_support) <= 64:
            raise ValueError("competence min_support must be within [2, 64]")
        if not 0.0 < float(self.min_consistency) <= 1.0:
            raise ValueError("competence min_consistency must be within (0, 1]")


DEFAULT_COMPETENCE_GATE = CompetenceGate()


@dataclass(slots=True)
class CompetenceEvidence:
    controller_seed_ref: str
    effect_evidence_refs: tuple[str, ...] = ()
    context_evidence_refs: tuple[str, ...] = ()
    controllability_evidence_refs: tuple[str, ...] = ()
    support: int = 0
    failures: int = 0
    reproducibility: float = 0.0
    controllability: float = 0.0
    directional_consistency: float = 0.0
    gate: CompetenceGate = DEFAULT_COMPETENCE_GATE

    @property
    def maturity(self) -> CompetenceMaturity:
        """Evidence projection only; no separately mutable maturity truth."""
        if (
            self.support < self.gate.min_support
            or self.controllability <= 0.002
            or self.reproducibility < (2.0 / 3.0)
            or self.directional_consistency < self.gate.min_consistency
        ):
            if self.support >= 2 and self.controllability > 0.0:
                return CompetenceMaturity.EMERGING
            return CompetenceMaturity.CANDIDATE
        if (
            self.support >= 8
            and self.reproducibility >= 0.85
            and self.directional_consistency >= 0.80
            and self.failures <= max(1, self.support // 8)
        ):
            return CompetenceMaturity.ROBUST
        return CompetenceMaturity.ESTABLISHED


@dataclass(slots=True)
class CompetenceCandidate:
    candidate_id: str
    controller_seed: object
    evidence: CompetenceEvidence
    created_tick: int
    last_supported_tick: int

    @property
    def maturity(self) -> CompetenceMaturity:
        return self.evidence.maturity


@dataclass(slots=True)
class MotorCompetence:
    competence_id: str
    controller_id: str
    effect_id: str | None
    evidence: CompetenceEvidence
    parent_competence_ids: tuple[str, ...] = ()
    controller_strategy_ref: str | None = None

    @property
    def maturity(self) -> CompetenceMaturity:
        return self.evidence.maturity


def _competence_evidence_payload(evidence: CompetenceEvidence) -> dict[str, object]:
    return {
        "controller_seed_ref": str(evidence.controller_seed_ref),
        "effect_evidence_refs": list(evidence.effect_evidence_refs),
        "context_evidence_refs": list(evidence.context_evidence_refs),
        "controllability_evidence_refs": list(evidence.controllability_evidence_refs),
        "support": int(evidence.support),
        "failures": int(evidence.failures),
        "reproducibility": float(evidence.reproducibility),
        "controllability": float(evidence.controllability),
        "directional_consistency": float(evidence.directional_consistency),
    }


def _restore_competence_evidence(
    payload: object, *, gate: CompetenceGate = DEFAULT_COMPETENCE_GATE
) -> CompetenceEvidence:
    if not isinstance(payload, dict):
        raise ValueError("invalid competence evidence payload")
    required = {
        "controller_seed_ref",
        "effect_evidence_refs",
        "context_evidence_refs",
        "controllability_evidence_refs",
        "support",
        "failures",
        "reproducibility",
        "controllability",
        "directional_consistency",
    }
    if set(payload) != required:
        raise ValueError("invalid competence evidence fields")
    controller_seed_ref = payload["controller_seed_ref"]
    if not isinstance(controller_seed_ref, str) or not controller_seed_ref:
        raise ValueError("invalid controller_seed_ref")
    for key in ("effect_evidence_refs", "context_evidence_refs", "controllability_evidence_refs"):
        val = payload[key]
        if not isinstance(val, list) or any(not isinstance(r, str) or not r for r in val):
            raise ValueError(f"invalid {key}")
    support = payload["support"]
    failures = payload["failures"]
    if isinstance(support, bool) or not isinstance(support, int) or support < 0:
        raise ValueError("invalid support")
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        raise ValueError("invalid failures")
    reproducibility = float(payload["reproducibility"])
    controllability = float(payload["controllability"])
    directional_consistency = float(payload["directional_consistency"])
    for name, val in (
        ("reproducibility", reproducibility),
        ("controllability", controllability),
        ("directional_consistency", directional_consistency),
    ):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    return CompetenceEvidence(
        controller_seed_ref=controller_seed_ref,
        effect_evidence_refs=tuple(payload["effect_evidence_refs"]),
        context_evidence_refs=tuple(payload["context_evidence_refs"]),
        controllability_evidence_refs=tuple(payload["controllability_evidence_refs"]),
        support=support,
        failures=failures,
        reproducibility=reproducibility,
        controllability=controllability,
        directional_consistency=directional_consistency,
        gate=gate,
    )


def _motor_competence_payload(competence: MotorCompetence) -> dict[str, object]:
    return {
        "competence_id": str(competence.competence_id),
        "controller_id": str(competence.controller_id),
        "effect_id": competence.effect_id,
        "evidence": _competence_evidence_payload(competence.evidence),
        "parent_competence_ids": list(competence.parent_competence_ids),
        "controller_strategy_ref": competence.controller_strategy_ref,
    }


def _restore_motor_competence(
    payload: object, *, gate: CompetenceGate = DEFAULT_COMPETENCE_GATE
) -> MotorCompetence:
    if not isinstance(payload, dict):
        raise ValueError("invalid motor competence payload")
    required = {
        "competence_id",
        "controller_id",
        "effect_id",
        "evidence",
        "parent_competence_ids",
        "controller_strategy_ref",
    }
    if set(payload) != required:
        raise ValueError("invalid motor competence fields")
    competence_id = payload["competence_id"]
    controller_id = payload["controller_id"]
    effect_id = payload["effect_id"]
    if not isinstance(competence_id, str) or not competence_id:
        raise ValueError("invalid competence_id")
    if not isinstance(controller_id, str) or not controller_id:
        raise ValueError("invalid controller_id")
    if effect_id is not None and (not isinstance(effect_id, str) or not effect_id):
        raise ValueError("invalid effect_id")
    parents = payload["parent_competence_ids"]
    if not isinstance(parents, list) or any(not isinstance(p, str) or not p for p in parents):
        raise ValueError("invalid parent_competence_ids")
    strategy_ref = payload["controller_strategy_ref"]
    if strategy_ref is not None and (not isinstance(strategy_ref, str) or not strategy_ref):
        raise ValueError("invalid controller_strategy_ref")
    evidence = _restore_competence_evidence(payload["evidence"], gate=gate)
    return MotorCompetence(
        competence_id=competence_id,
        controller_id=controller_id,
        effect_id=effect_id,
        evidence=evidence,
        parent_competence_ids=tuple(parents),
        controller_strategy_ref=strategy_ref,
    )


class CompetenceLibrary:
    SCHEMA_VERSION = 1

    def __init__(self, *, max_competences: int = 512) -> None:
        if (
            isinstance(max_competences, bool)
            or not isinstance(max_competences, int)
            or max_competences < 1
        ):
            raise ValueError("max_competences must be positive")
        self._max = int(max_competences)
        self._items: dict[str, MotorCompetence] = {}

    def add(self, competence: MotorCompetence) -> None:
        self._items[competence.competence_id] = competence
        if len(self._items) > self._max:
            retained = sorted(
                self._items.values(),
                key=lambda item: (
                    -item.evidence.support,
                    item.evidence.failures,
                    item.competence_id,
                ),
            )[: self._max]
            self._items = {item.competence_id: item for item in retained}

    def get(self, competence_id: str) -> MotorCompetence | None:
        return self._items.get(competence_id)

    def for_effect(self, effect_id: str) -> tuple[MotorCompetence, ...]:
        result = [item for item in self._items.values() if item.effect_id == effect_id]
        return tuple(
            sorted(
                result,
                key=lambda item: (-item.evidence.support, item.competence_id),
            )
        )

    @property
    def items(self) -> tuple[MotorCompetence, ...]:
        return tuple(sorted(self._items.values(), key=lambda item: item.competence_id))

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_competences": self._max,
            "competences": [
                _motor_competence_payload(item) for item in self.items
            ],
        }

    def to_bytes(self) -> bytes:
        """Return a deterministic, self-contained export for owner checkpoints."""
        return json.dumps(
            self.checkpoint(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        gate: CompetenceGate = DEFAULT_COMPETENCE_GATE,
    ) -> "CompetenceLibrary":
        """Restore only an unambiguous JSON export with the current schema."""
        if not isinstance(payload, bytes):
            raise ValueError("competence library export must be bytes")

        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate competence library field")
                result[key] = value
            return result

        try:
            decoded = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid competence library export") from exc
        return cls.restore(decoded, gate=gate)

    @classmethod
    def restore(
        cls,
        payload: object,
        *,
        gate: CompetenceGate = DEFAULT_COMPETENCE_GATE,
    ) -> "CompetenceLibrary":
        if not isinstance(payload, dict):
            raise ValueError("invalid competence library checkpoint")
        expected_fields = {"schema_version", "max_competences", "competences"}
        if set(payload) != expected_fields:
            raise ValueError("invalid competence library checkpoint fields")
        version = payload["schema_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != cls.SCHEMA_VERSION
        ):
            raise ValueError("unsupported competence library checkpoint version")
        max_competences = payload["max_competences"]
        if (
            isinstance(max_competences, bool)
            or not isinstance(max_competences, int)
            or max_competences < 1
        ):
            raise ValueError("invalid competence library capacity")
        raw_items = payload["competences"]
        if not isinstance(raw_items, list) or len(raw_items) > max_competences:
            raise ValueError("invalid or unbounded competence library entries")
        library = cls(max_competences=max_competences)
        for raw_item in raw_items:
            competence = _restore_motor_competence(raw_item, gate=gate)
            if competence.competence_id in library._items:
                raise ValueError("duplicate competence_id in checkpoint")
            library._items[competence.competence_id] = competence
        return library
