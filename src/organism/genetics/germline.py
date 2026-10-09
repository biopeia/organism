"""Genetic and explicitly optional epigenetic inheritance for Genome v2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .genome import Genome, flatten_genes
from .schema import DEFAULT_GENOME_SCHEMA, GenomeSchema

PERSISTENCE_TICKS = 64


@dataclass(frozen=True, slots=True)
class EpigeneticMark:
    locus: str
    delta: float
    strength: float = 1.0
    generations_left: int = 3

    def __post_init__(self) -> None:
        if not self.locus:
            raise ValueError("epigenetic locus must not be empty")
        if not math.isfinite(self.delta):
            raise ValueError("epigenetic delta must be finite")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("epigenetic strength must be in [0,1]")
        if self.generations_left < 0:
            raise ValueError("generations_left must be non-negative")

    def decay(self, fraction: float) -> "EpigeneticMark | None":
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("epigenetic decay must be in [0,1]")
        strength = self.strength * (1.0 - fraction)
        remaining = self.generations_left - 1
        if strength < 0.05 or remaining <= 0:
            return None
        return EpigeneticMark(
            locus=self.locus,
            delta=self.delta,
            strength=strength,
            generations_left=remaining,
        )


@dataclass(frozen=True, slots=True)
class EpigeneticProtocol:
    """Optional transgenerational extension, outside the baseline genotype."""

    enabled: bool = False
    acquired_capture_enabled: bool = False
    decay: float = 0.2
    max_marks: int = 8
    min_capture_delta: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(
            self.acquired_capture_enabled, bool
        ):
            raise ValueError("epigenetic protocol flags must be booleans")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("epigenetic decay must be in [0,1]")
        if self.max_marks < 1:
            raise ValueError("max_marks must be positive")
        if not math.isfinite(self.min_capture_delta) or self.min_capture_delta < 0.0:
            raise ValueError("min_capture_delta must be finite and non-negative")


@dataclass(slots=True)
class GermlineState:
    """Lifetime germline metadata; acquired capture is disabled by default."""

    birth_expression: dict[str, float | int]
    inherited_marks: dict[str, EpigeneticMark] = field(default_factory=dict)
    acquired_marks: dict[str, EpigeneticMark] = field(default_factory=dict)
    acquired_capture_enabled: bool = False
    capture_streaks: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_genome(
        cls,
        genome: Genome,
        *,
        inherited_marks: tuple[EpigeneticMark, ...] = (),
        acquired_capture_enabled: bool = False,
        schema: GenomeSchema = DEFAULT_GENOME_SCHEMA,
    ) -> "GermlineState":
        # Birth expression is the effective phenotype at birth, not the raw
        # genotype. Inherited marks must not later look like a lifetime
        # acquired deviation and be captured a second time.
        birth_expression = dict(flatten_genes(genome))
        for mark in inherited_marks:
            base = birth_expression.get(mark.locus)
            if base is None:
                raise ValueError(f"inherited mark targets unknown locus {mark.locus!r}")
            spec = schema.spec(mark.locus)
            if not spec.regulable:
                raise ValueError(f"inherited mark targets non-regulable locus {mark.locus!r}")
            birth_expression[mark.locus] = spec.clamp(float(base) + mark.delta * mark.strength)
        return cls(
            birth_expression=birth_expression,
            inherited_marks={mark.locus: mark for mark in inherited_marks},
            acquired_marks={},
            acquired_capture_enabled=acquired_capture_enabled,
        )

    def effective_value(
        self,
        genome: Genome,
        locus: str,
        *,
        schema: GenomeSchema = DEFAULT_GENOME_SCHEMA,
        include_acquired_marks: bool = False,
    ) -> float | int:
        """Return the phenotype expressed from inherited marks at birth.

        Acquired marks are records for possible transmission, not an extra
        control signal for the living individual. They enter phenotype only
        after being passed to a child's ``inherited_marks``.
        """
        base = flatten_genes(genome)[locus]
        spec = schema.spec(locus)
        value = float(base)
        mark = self.inherited_marks.get(locus)
        if mark is not None:
            value += mark.delta * mark.strength
        if include_acquired_marks:
            acquired = self.acquired_marks.get(locus)
            if acquired is not None:
                value += acquired.delta * acquired.strength
        return spec.clamp(value)

    def capture_acquired_variation(
        self,
        current_expression: Mapping[str, float],
        *,
        genome: Genome,
        schema: GenomeSchema = DEFAULT_GENOME_SCHEMA,
        protocol: EpigeneticProtocol | None = None,
    ) -> tuple[str, ...]:
        """Capture only under an explicitly enabled transgenerational protocol."""
        active = protocol or EpigeneticProtocol()
        if not (
            self.acquired_capture_enabled and active.enabled and active.acquired_capture_enabled
        ):
            self.capture_streaks.clear()
            return ()
        captured: list[str] = []
        max_marks = active.max_marks
        observed_loci = set(current_expression)
        for locus in tuple(self.capture_streaks):
            if locus not in observed_loci:
                self.capture_streaks.pop(locus, None)
        for locus, current in sorted(current_expression.items()):
            if locus in self.acquired_marks:
                continue
            if len(self.acquired_marks) >= max_marks and locus not in self.acquired_marks:
                break
            spec = schema.spec(locus)
            if not spec.regulable:
                continue
            birth = float(self.birth_expression.get(locus, flatten_genes(genome)[locus]))
            delta = float(current) - birth
            if abs(delta) < active.min_capture_delta:
                self.capture_streaks.pop(locus, None)
                continue
            streak = self.capture_streaks.get(locus, 0) + 1
            self.capture_streaks[locus] = streak
            if streak < PERSISTENCE_TICKS:
                continue
            self.acquired_marks[locus] = EpigeneticMark(locus=locus, delta=delta)
            self.capture_streaks.pop(locus, None)
            captured.append(locus)
        return tuple(captured)


@dataclass(frozen=True, slots=True)
class InheritancePackage:
    genome: Genome
    epigenetic_marks: tuple[EpigeneticMark, ...]
    parent_ids: tuple[str, ...]
    generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.genome, Genome):
            raise ValueError("inheritance package requires a Genome v2 genome")
        if (
            not isinstance(self.parent_ids, tuple)
            or not self.parent_ids
            or any(not isinstance(parent_id, str) or not parent_id for parent_id in self.parent_ids)
        ):
            raise ValueError("inheritance package requires non-empty parent identifiers")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("inheritance package parent identifiers must be unique")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise ValueError("offspring generation must be an integer")
        if self.generation < 1:
            raise ValueError("offspring generation must be at least 1")
        if not isinstance(self.epigenetic_marks, tuple) or any(
            not isinstance(mark, EpigeneticMark) for mark in self.epigenetic_marks
        ):
            raise ValueError("inheritance package marks must be EpigeneticMark values")
        loci = [mark.locus for mark in self.epigenetic_marks]
        if len(set(loci)) != len(loci):
            raise ValueError("inheritance package marks must have unique loci")


__all__ = [
    "EpigeneticMark",
    "EpigeneticProtocol",
    "GermlineState",
    "InheritancePackage",
    "PERSISTENCE_TICKS",
]
