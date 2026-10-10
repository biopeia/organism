"""Organism-owned sensorimotor effect space.

No body, world, lab or evaluator semantic label is accepted here.  Effects are
identified only from opaque organism-visible feature references and observed
transition structure.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..capacity import CapacityPressure

_ALLOWED_PREFIXES = ("signal.", "latent.", "part.", "channel.", "internal.", "effect.")
_EFFECT_SPACE_FIELDS = {
    "schema_version",
    "max_effects",
    "support",
    "footprints",
    "capacity_pressure",
}
_SUPPORT_FIELDS = {"signature", "count"}
_FOOTPRINT_FIELDS = {"effect_id", "support", "atoms"}


def _checkpoint_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid effect-space {field}")
    return value


def _valid_feature_ref(value: str) -> bool:
    return bool(value) and value.startswith(_ALLOWED_PREFIXES)


@dataclass(frozen=True, slots=True)
class EffectRepresentation:
    effect_id: str
    feature_refs: tuple[str, ...]
    transition_signature: tuple[tuple[str, int], ...]
    support: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.effect_id.startswith("effect."):
            raise ValueError("effect_id must be opaque and organism-owned")
        if not self.feature_refs or any(not _valid_feature_ref(item) for item in self.feature_refs):
            raise ValueError("effect features must be organism-owned references")
        if self.support < 1 or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("invalid effect evidence")


@dataclass(frozen=True, slots=True)
class EffectTarget:
    effect_id: str
    desired_change: float | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.effect_id.startswith("effect."):
            raise ValueError("EffectTarget must reference organism-owned EffectSpace")
        if self.desired_change is not None and not math.isfinite(float(self.desired_change)):
            raise ValueError("desired_change must be finite")
        if self.tolerance is not None and (
            not math.isfinite(float(self.tolerance)) or self.tolerance < 0
        ):
            raise ValueError("tolerance must be finite and non-negative")


class EffectMatcher:
    """Effect equivalence (§33).

    v1 equivalence is identity: same organism-owned effect -> 1.0, otherwise
    0.0.  All intent/feedback logic compares effects through this class so a
    graded similarity can replace identity without touching its callers.
    """

    def similarity(
        self,
        expected: EffectRepresentation,
        observed: EffectRepresentation,
    ) -> float:
        return 1.0 if expected.effect_id == observed.effect_id else 0.0

    def match(
        self,
        space: "EffectSpace",
        *,
        expected_effect_id: str | None,
        observed_effect_id: str | None,
    ) -> float:
        """Similarity of two effects known to ``space``; unknown/absent -> 0.0."""
        if expected_effect_id is None or observed_effect_id is None:
            return 0.0
        expected = space.get(expected_effect_id)
        observed = space.get(observed_effect_id)
        if expected is None or observed is None:
            return 0.0
        return self.similarity(expected, observed)


# -- Factorized Effect Representation v1 §4.1: atomic effects --------------------
MAX_ATOMS_PER_TRANSITION = 16


def magnitude_class(bucket: int) -> int:
    """Coarse magnitude of a v4 bucket (spec §4.1): exact, fixed mapping."""
    size = abs(int(bucket))
    if not 1 <= size <= 7:
        raise ValueError("a magnitude class needs a non-zero bucket within [-7, 7]")
    return 1 if size <= 2 else (2 if size <= 4 else 3)


@dataclass(frozen=True, slots=True)
class EffectAtom:
    """One organism feature moving in one direction by a coarse amount."""

    feature_ref: str
    direction: int
    magnitude_class: int

    def __post_init__(self) -> None:
        if not _valid_feature_ref(self.feature_ref):
            raise ValueError("atom features must be organism-owned references")
        if self.direction not in (-1, 1):
            raise ValueError("atom direction must be -1 or +1")
        if self.magnitude_class not in (1, 2, 3):
            raise ValueError("atom magnitude class must be 1, 2 or 3")

    @property
    def effect_id(self) -> str:
        material = f"atom|{self.feature_ref}|{self.direction}|{self.magnitude_class}"
        return "effect.atom." + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def from_bucket(cls, feature_ref: str, bucket: int) -> "EffectAtom":
        return cls(str(feature_ref), 1 if bucket > 0 else -1, magnitude_class(bucket))


def atom_change_key(atom: EffectAtom) -> str:
    """The change an atom denotes, independent of its magnitude class."""
    return f"{atom.feature_ref}|{atom.direction:+d}"


def atoms_from_signature(
    signature: tuple[tuple[str, int], ...],
) -> tuple[EffectAtom, ...]:
    """Exact decomposition of a v4 whole-state signature (spec §8)."""
    return tuple(
        sorted(
            (EffectAtom.from_bucket(feature, bucket) for feature, bucket in signature if bucket),
            key=lambda atom: atom.feature_ref,
        )
    )


def atoms_from_changes(
    changes: Mapping[str, float],
    *,
    limit: int = MAX_ATOMS_PER_TRANSITION,
) -> tuple[EffectAtom, ...]:
    """The atoms of one transition, bounded: largest magnitudes first."""
    return bounded_atoms(EffectSpace.signature(changes), limit=limit)


def bounded_atoms(
    signature: tuple[tuple[str, int], ...],
    *,
    limit: int = MAX_ATOMS_PER_TRANSITION,
) -> tuple[EffectAtom, ...]:
    """A signature's atoms under the per-transition bound: largest changes first."""
    if limit < 1:
        raise ValueError("limit must be positive")
    kept = sorted(signature, key=lambda item: (-abs(item[1]), item[0]))[:limit]
    return atoms_from_signature(tuple(kept))


class EffectSpace:
    """Bounded registry of recurring opaque transition signatures."""

    SCHEMA_VERSION = 1

    def __init__(self, *, max_effects: int = 512) -> None:
        if isinstance(max_effects, bool) or not isinstance(max_effects, int) or max_effects < 1:
            raise ValueError("max_effects must be positive")
        self._max_effects = max_effects
        self._support: dict[tuple[tuple[str, int], ...], int] = {}
        self._effects: dict[str, EffectRepresentation] = {}
        # Factorized Effect Representation v1 §14.1: footprint effects are
        # keyed by entity, registered explicitly and never evicted by the
        # whole-state bound (which saturates in high-dimensional bodies).
        self._footprints: dict[str, tuple[EffectRepresentation, tuple[EffectAtom, ...]]] = {}
        # Wave 0 measurement only (Cross-Domain Revision Coherence v1 §2.1).
        self.pressure = CapacityPressure(self._max_effects)

    @staticmethod
    def signature(changes: Mapping[str, float]) -> tuple[tuple[str, int], ...]:
        quantized: list[tuple[str, int]] = []
        for feature, value in sorted(changes.items()):
            if not _valid_feature_ref(str(feature)):
                continue
            number = float(value)
            if not math.isfinite(number) or abs(number) < 1e-9:
                continue
            bucket = max(-7, min(7, round(number * 7.0)))
            if bucket:
                quantized.append((str(feature), bucket))
        return tuple(quantized)

    @staticmethod
    def _id(signature: tuple[tuple[str, int], ...]) -> str:
        material = repr(signature).encode("utf-8")
        return "effect." + hashlib.sha256(material).hexdigest()[:24]

    def observe(self, changes: Mapping[str, float]) -> EffectRepresentation | None:
        signature = self.signature(changes)
        if not signature:
            return None
        support = self._support.get(signature, 0) + 1
        self._support[signature] = support
        effect_id = self._id(signature)
        confidence = min(1.0, support / 8.0)
        admitted = effect_id not in self._effects
        representation = EffectRepresentation(
            effect_id=effect_id,
            feature_refs=tuple(feature for feature, _ in signature),
            transition_signature=signature,
            support=support,
            confidence=confidence,
        )
        self._effects[effect_id] = representation
        if admitted:
            self.pressure.note_admitted(effect_id)
        if len(self._effects) > self._max_effects:
            before = tuple(self._effects)
            retained = sorted(
                self._effects.values(),
                key=lambda item: (-item.support, -item.confidence, item.effect_id),
            )[: self._max_effects]
            self._effects = {item.effect_id: item for item in retained}
            retained_signatures = {item.transition_signature for item in retained}
            self._support = {
                key: value for key, value in self._support.items() if key in retained_signatures
            }
            self.pressure.note_evicted(ref for ref in before if ref not in self._effects)
        return representation

    def register_footprint(
        self, effect_id: str, atoms: tuple[EffectAtom, ...], *, support: int
    ) -> EffectRepresentation:
        """(Re)register a footprint entity's effect with its current atoms."""
        if not effect_id.startswith("effect.entity."):
            raise ValueError("footprint effects are keyed by footprint entity")
        ordered = tuple(sorted(set(atoms), key=lambda atom: atom.effect_id))
        if not ordered or support < 1:
            raise ValueError("a footprint effect needs atoms and support")
        representation = EffectRepresentation(
            effect_id=effect_id,
            feature_refs=tuple(sorted({atom.feature_ref for atom in ordered})),
            transition_signature=tuple(
                sorted(
                    (atom.feature_ref, atom.direction * atom.magnitude_class) for atom in ordered
                )
            ),
            support=int(support),
            confidence=min(1.0, support / 8.0),
        )
        self._footprints[effect_id] = (representation, ordered)
        return representation

    def footprint_atoms(self, effect_id: str) -> tuple[str, ...] | None:
        """Atom effect ids of a registered footprint effect."""
        entry = self._footprints.get(effect_id)
        return tuple(atom.effect_id for atom in entry[1]) if entry is not None else None

    def footprint_change_keys(self, effect_id: str) -> tuple[str, ...] | None:
        """(feature, direction) keys of a footprint effect: what "the intended
        change happened" is judged on; magnitude class is detail (§14.1)."""
        entry = self._footprints.get(effect_id)
        if entry is None:
            return None
        return tuple(sorted({atom_change_key(atom) for atom in entry[1]}))

    def get(self, effect_id: str) -> EffectRepresentation | None:
        found = self._effects.get(effect_id)
        if found is not None:
            return found
        entry = self._footprints.get(effect_id)
        return entry[0] if entry is not None else None

    @property
    def effects(self) -> tuple[EffectRepresentation, ...]:
        return tuple(
            sorted(
                (*self._effects.values(), *(entry[0] for entry in self._footprints.values())),
                key=lambda item: item.effect_id,
            )
        )

    def target(
        self, effect_id: str, *, desired_change: float | None = None, tolerance: float | None = None
    ) -> EffectTarget:
        if self.get(effect_id) is None:
            raise KeyError("EffectTarget cannot be constructed from an external/unknown effect")
        return EffectTarget(effect_id, desired_change, tolerance)

    def capacity_snapshot(self) -> dict[str, object]:
        """Pressure on the bounded whole-state effects (footprints are exempt)."""
        return self.pressure.snapshot(len(self._effects))

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_effects": self._max_effects,
            "support": [
                {"signature": [[key, value] for key, value in signature], "count": count}
                for signature, count in sorted(self._support.items())
            ],
            "footprints": [
                {
                    "effect_id": effect_id,
                    "support": representation.support,
                    "atoms": [
                        [atom.feature_ref, atom.direction, atom.magnitude_class] for atom in atoms
                    ],
                }
                for effect_id, (representation, atoms) in sorted(self._footprints.items())
            ],
            "capacity_pressure": self.pressure.checkpoint(),
        }

    @classmethod
    def restore(cls, payload: object) -> "EffectSpace":
        if not isinstance(payload, Mapping) or set(payload) != _EFFECT_SPACE_FIELDS:
            raise ValueError("invalid effect-space checkpoint fields")
        version = payload["schema_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != cls.SCHEMA_VERSION
        ):
            raise ValueError("unsupported effect-space checkpoint")
        max_effects = _checkpoint_int(payload["max_effects"], field="capacity", minimum=1)
        raw_support = payload["support"]
        raw_footprints = payload["footprints"]
        if (
            not isinstance(raw_support, list)
            or len(raw_support) > max_effects
            or not isinstance(raw_footprints, list)
        ):
            raise ValueError("invalid effect-space checkpoint collections")

        obj = cls(max_effects=max_effects)
        for item in raw_support:
            if not isinstance(item, Mapping) or set(item) != _SUPPORT_FIELDS:
                raise ValueError("invalid effect-space support")
            raw_signature = item["signature"]
            if not isinstance(raw_signature, list) or not raw_signature:
                raise ValueError("invalid effect-space signature")
            signature_items: list[tuple[str, int]] = []
            seen_features: set[str] = set()
            for pair in raw_signature:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("invalid effect-space signature entry")
                feature, bucket = pair
                if not isinstance(feature, str) or not _valid_feature_ref(feature):
                    raise ValueError("invalid effect-space feature reference")
                bucket = _checkpoint_int(bucket, field="signature bucket", minimum=-7)
                if bucket == 0 or bucket > 7:
                    raise ValueError("invalid effect-space signature bucket")
                if feature in seen_features:
                    raise ValueError("duplicate effect-space signature feature")
                seen_features.add(feature)
                signature_items.append((feature, bucket))
            signature = tuple(signature_items)
            if tuple(sorted(signature)) != signature:
                raise ValueError("effect-space signature must be sorted")
            count = _checkpoint_int(item["count"], field="support count", minimum=1)
            if signature in obj._support:
                raise ValueError("duplicate effect-space support")
            obj._support[signature] = count
            effect_id = obj._id(signature)
            obj._effects[effect_id] = EffectRepresentation(
                effect_id=effect_id,
                feature_refs=tuple(key for key, _ in signature),
                transition_signature=signature,
                support=count,
                confidence=min(1.0, count / 8.0),
            )
        seen_footprints: set[str] = set()
        for item in raw_footprints:
            if not isinstance(item, Mapping) or set(item) != _FOOTPRINT_FIELDS:
                raise ValueError("invalid effect-space footprint")
            effect_id = item["effect_id"]
            if not isinstance(effect_id, str) or not effect_id.startswith("effect.entity."):
                raise ValueError("invalid effect-space footprint id")
            if effect_id in seen_footprints:
                raise ValueError("duplicate effect-space footprint")
            seen_footprints.add(effect_id)
            support = _checkpoint_int(item["support"], field="footprint support", minimum=1)
            raw_atoms = item["atoms"]
            if not isinstance(raw_atoms, list) or not raw_atoms:
                raise ValueError("invalid effect-space footprint atoms")
            atoms: list[EffectAtom] = []
            for atom in raw_atoms:
                if not isinstance(atom, list) or len(atom) != 3:
                    raise ValueError("invalid effect-space footprint atom")
                feature, direction, magnitude = atom
                if not isinstance(feature, str):
                    raise ValueError("invalid effect-space atom feature")
                direction = _checkpoint_int(direction, field="atom direction", minimum=-1)
                magnitude = _checkpoint_int(magnitude, field="atom magnitude", minimum=1)
                atoms.append(EffectAtom(feature, direction, magnitude))
            if len(set(atoms)) != len(atoms):
                raise ValueError("duplicate effect-space footprint atom")
            obj.register_footprint(effect_id, tuple(atoms), support=support)
        obj.pressure = CapacityPressure.restore(
            payload["capacity_pressure"], capacity=obj._max_effects
        )
        return obj
