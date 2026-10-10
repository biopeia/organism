"""Evidence-first sequential competence composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


_CHECKPOINT_FIELDS = {"schema_version", "max_relations", "sequential"}
_EVIDENCE_FIELDS = {
    "first_competence_id",
    "second_competence_id",
    "effect_id",
    "support",
    "failures",
}


def _require_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(slots=True)
class SequentialCompositionEvidence:
    first_competence_id: str
    second_competence_id: str
    effect_id: str
    support: int = 0
    failures: int = 0

    @property
    def observations(self) -> int:
        return self.support + self.failures

    @property
    def reproducibility(self) -> float:
        return self.support / max(1, self.observations)

    @property
    def established(self) -> bool:
        return self.support >= 4 and self.reproducibility >= (2.0 / 3.0)

    def checkpoint(self) -> dict[str, object]:
        return {
            "first_competence_id": self.first_competence_id,
            "second_competence_id": self.second_competence_id,
            "effect_id": self.effect_id,
            "support": self.support,
            "failures": self.failures,
        }


class CompositionEngine:
    SCHEMA_VERSION = 1

    def __init__(self, *, max_relations: int = 512) -> None:
        if (
            isinstance(max_relations, bool)
            or not isinstance(max_relations, int)
            or max_relations < 1
        ):
            raise ValueError("max_relations must be positive")
        self._max_relations = max_relations
        self._sequential: dict[
            tuple[str, str, str],
            SequentialCompositionEvidence,
        ] = {}

    def observe(
        self,
        first: str,
        second: str,
        effect_id: str,
        *,
        success: bool,
    ) -> SequentialCompositionEvidence:
        if not first or not second or first == second:
            raise ValueError("sequential composition requires two distinct competences")
        if not effect_id.startswith("effect."):
            raise ValueError("composition effect must be organism-owned")
        key = (first, second, effect_id)
        evidence = self._sequential.setdefault(
            key,
            SequentialCompositionEvidence(first, second, effect_id),
        )
        if success:
            evidence.support += 1
        else:
            evidence.failures += 1
        if len(self._sequential) > self._max_relations:
            retained = sorted(
                self._sequential.items(),
                key=lambda item: (
                    -item[1].support,
                    item[1].failures,
                    item[0],
                ),
            )[: self._max_relations]
            self._sequential = dict(retained)
        return evidence

    def observe_absence(self, first: str, second: str) -> None:
        """Record that an observed A->B completion failed to reproduce known effects."""
        for key, evidence in self._sequential.items():
            if key[0] == first and key[1] == second:
                evidence.failures += 1

    @property
    def evidence(self) -> tuple[SequentialCompositionEvidence, ...]:
        return tuple(
            sorted(
                self._sequential.values(),
                key=lambda item: (
                    item.first_competence_id,
                    item.second_competence_id,
                    item.effect_id,
                ),
            )
        )

    @property
    def established(self) -> tuple[SequentialCompositionEvidence, ...]:
        return tuple(item for item in self.evidence if item.established)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_relations": self._max_relations,
            "sequential": [item.checkpoint() for item in self.evidence],
        }

    @classmethod
    def restore(cls, payload: object) -> "CompositionEngine":
        if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_FIELDS:
            raise ValueError("invalid composition checkpoint fields")
        version = payload["schema_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != cls.SCHEMA_VERSION
        ):
            raise ValueError("unsupported composition checkpoint")
        max_relations = payload["max_relations"]
        if (
            isinstance(max_relations, bool)
            or not isinstance(max_relations, int)
            or max_relations < 1
        ):
            raise ValueError("invalid composition relation capacity")
        raw = payload["sequential"]
        if not isinstance(raw, list) or len(raw) > max_relations:
            raise ValueError("invalid composition checkpoint")
        engine = cls(max_relations=max_relations)
        for item in raw:
            if not isinstance(item, Mapping) or set(item) != _EVIDENCE_FIELDS:
                raise ValueError("invalid composition evidence")
            evidence = SequentialCompositionEvidence(
                first_competence_id=_require_identifier(
                    item["first_competence_id"], field="first_competence_id"
                ),
                second_competence_id=_require_identifier(
                    item["second_competence_id"], field="second_competence_id"
                ),
                effect_id=_require_identifier(item["effect_id"], field="effect_id"),
                support=_require_non_negative_int(item["support"], field="support"),
                failures=_require_non_negative_int(
                    item["failures"], field="failures"
                ),
            )
            if (
                evidence.first_competence_id == evidence.second_competence_id
                or not evidence.effect_id.startswith("effect.")
            ):
                raise ValueError("invalid composition evidence")
            key = (
                evidence.first_competence_id,
                evidence.second_competence_id,
                evidence.effect_id,
            )
            if key in engine._sequential:
                raise ValueError("duplicate composition evidence")
            engine._sequential[key] = evidence
        return engine
