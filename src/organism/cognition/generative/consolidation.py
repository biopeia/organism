"""Generative-use accounting kept separate from factual representation counts."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ...capacity import CapacityPressure
from .types import bounded_identifier, bounded_tuple, unit_interval


def _checkpoint_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _checkpoint_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


@dataclass(frozen=True, slots=True)
class GenerativeConsolidationSignal:
    recurrent_activation: int
    cross_episode_reuse: int
    hypothesis_persistence: int
    model_disagreement: float
    generative_demand: float
    independent_episode_count: int
    source_diversity: int

    def __post_init__(self) -> None:
        for name, value in (
            ("recurrent_activation", self.recurrent_activation),
            ("cross_episode_reuse", self.cross_episode_reuse),
            ("hypothesis_persistence", self.hypothesis_persistence),
            ("independent_episode_count", self.independent_episode_count),
            ("source_diversity", self.source_diversity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        unit_interval(self.model_disagreement, name="model_disagreement")
        unit_interval(self.generative_demand, name="generative_demand")


@dataclass(frozen=True, slots=True)
class GenerativeCandidateProjection:
    """Producer-neutral request for the existing structural contention layer."""

    candidate_id: str
    family: str
    producer_id: str
    eligible_tick: int
    mutation_payloads: tuple[str, ...]

    def __post_init__(self) -> None:
        bounded_identifier(self.candidate_id, name="candidate_id")
        bounded_identifier(self.family, name="family")
        bounded_identifier(self.producer_id, name="producer_id")
        if (
            isinstance(self.eligible_tick, bool)
            or not isinstance(self.eligible_tick, int)
            or self.eligible_tick < 0
        ):
            raise ValueError("eligible_tick must be a non-negative integer")
        bounded_tuple(self.mutation_payloads, name="mutation_payloads")
        if not self.mutation_payloads:
            raise ValueError("mutation_payloads must not be empty")


class GenerativeUseTracker:
    """Bounded generated-use aggregates kept apart from factual observations."""

    MAX_REPRESENTATIONS = 256
    MAX_EPISODES_PER_REPRESENTATION = 64
    MAX_SOURCES_PER_REPRESENTATION = 64

    def __init__(self) -> None:
        self._activation_counts: dict[str, int] = {}
        self._episodes: dict[str, set[str]] = {}
        self._sources: dict[str, set[str]] = {}
        self._hypothesis_counts: dict[str, int] = {}
        self._disagreement_sum: dict[str, float] = {}
        self._disagreement_count: dict[str, int] = {}
        self._last_use_organism_step: dict[str, int] = {}
        # Wave 0 measurement only (Cross-Domain Revision Coherence v1 §2.1):
        # eviction pressure, and reactivations that bring no new context
        # (episode) and no new source/branch ref. Never read by decisions.
        self.pressure = CapacityPressure(self.MAX_REPRESENTATIONS)
        self.sterile_reactivations: dict[str, int] = {}

    def record(
        self,
        *,
        representation_ref: str,
        episode_id: str,
        state_id: str,
        model_ids: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        hypothesis_ref: str | None = None,
        model_disagreement: float = 0.0,
        organism_step: int | None = None,
    ) -> None:
        bounded_identifier(representation_ref, name="representation_ref")
        bounded_identifier(episode_id, name="episode_id")
        bounded_identifier(state_id, name="state_id")
        if not isinstance(model_ids, tuple) or not isinstance(source_refs, tuple):
            raise ValueError("model_ids and source_refs must be tuples")
        for model_id in model_ids:
            bounded_identifier(model_id, name="model_id")
        for source_ref in source_refs:
            bounded_identifier(source_ref, name="source_ref")
        if hypothesis_ref is not None:
            bounded_identifier(hypothesis_ref, name="hypothesis_ref")
        unit_interval(model_disagreement, name="model_disagreement")
        if organism_step is not None and (isinstance(organism_step, bool) or not isinstance(organism_step, int) or organism_step < 0):
            raise ValueError("organism_step must be a non-negative integer or None")

        if (
            representation_ref not in self._activation_counts
            and len(self._activation_counts) >= self.MAX_REPRESENTATIONS
        ):
            victim = next(iter(self._activation_counts))
            self._drop(victim)
            self.pressure.note_evicted((victim,))

        known = representation_ref in self._activation_counts
        if not known:
            self.pressure.note_admitted(representation_ref)
        seen_episode = episode_id in self._episodes.get(representation_ref, ())
        sources_before = len(self._sources.get(representation_ref, ()))
        self._activation_counts[representation_ref] = min(
            1_000_000,
            self._activation_counts.get(representation_ref, 0) + 1,
        )
        episodes = self._episodes.setdefault(representation_ref, set())
        if len(episodes) < self.MAX_EPISODES_PER_REPRESENTATION:
            episodes.add(episode_id)
        sources = self._sources.setdefault(representation_ref, set())
        for source_ref in source_refs:
            if len(sources) >= self.MAX_SOURCES_PER_REPRESENTATION:
                break
            sources.add(source_ref)
        if known and seen_episode and len(sources) == sources_before:
            self.sterile_reactivations[representation_ref] = (
                self.sterile_reactivations.get(representation_ref, 0) + 1
            )
        if hypothesis_ref is not None:
            self._hypothesis_counts[representation_ref] = min(
                1_000_000,
                self._hypothesis_counts.get(representation_ref, 0) + 1,
            )
        self._disagreement_sum[representation_ref] = min(
            1_000_000.0,
            self._disagreement_sum.get(representation_ref, 0.0) + model_disagreement,
        )
        self._disagreement_count[representation_ref] = min(
            1_000_000,
            self._disagreement_count.get(representation_ref, 0) + 1,
        )
        if organism_step is not None:
            self._last_use_organism_step[representation_ref] = max(
                organism_step,
                self._last_use_organism_step.get(representation_ref, organism_step),
            )

    def note_factual_sources(
        self,
        *,
        representation_ref: str,
        source_refs: tuple[str, ...],
    ) -> None:
        """Attach independently observed provenance after reconciliation.

        This does not increment activation or factual observation counts.  It
        only records which factual sources have later supported/contradicted a
        representation that was already used generatively.
        """
        bounded_identifier(representation_ref, name="representation_ref")
        if not isinstance(source_refs, tuple):
            raise ValueError("source_refs must be a tuple")
        sources = self._sources.setdefault(representation_ref, set())
        for source_ref in source_refs:
            bounded_identifier(source_ref, name="source_ref")
            if len(sources) >= self.MAX_SOURCES_PER_REPRESENTATION:
                break
            sources.add(source_ref)

    def signal(
        self, *, representation_ref: str, generative_demand: float
    ) -> GenerativeConsolidationSignal:
        bounded_identifier(representation_ref, name="representation_ref")
        unit_interval(generative_demand, name="generative_demand")
        count = self._activation_counts.get(representation_ref, 0)
        episodes = self._episodes.get(representation_ref, set())
        disagreement_count = self._disagreement_count.get(representation_ref, 0)
        return GenerativeConsolidationSignal(
            recurrent_activation=count,
            cross_episode_reuse=max(0, len(episodes) - 1),
            hypothesis_persistence=self._hypothesis_counts.get(representation_ref, 0),
            model_disagreement=(
                self._disagreement_sum.get(representation_ref, 0.0) / disagreement_count
                if disagreement_count
                else 0.0
            ),
            generative_demand=generative_demand,
            independent_episode_count=len(episodes),
            source_diversity=len(self._sources.get(representation_ref, set())),
        )

    @property
    def representation_refs(self) -> tuple[str, ...]:
        """Bounded durable representation ids with generative-use history."""
        return tuple(sorted(self._activation_counts))

    def last_use_organism_step(self, representation_ref: str) -> int | None:
        bounded_identifier(representation_ref, name="representation_ref")
        return self._last_use_organism_step.get(representation_ref)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "representations": [
                {
                    "representation_ref": representation_ref,
                    "activation_count": self._activation_counts[representation_ref],
                    "episode_ids": sorted(self._episodes.get(representation_ref, set())),
                    "source_refs": sorted(self._sources.get(representation_ref, set())),
                    "hypothesis_count": self._hypothesis_counts.get(representation_ref, 0),
                    "disagreement_sum": self._disagreement_sum.get(representation_ref, 0.0),
                    "disagreement_count": self._disagreement_count.get(representation_ref, 0),
                    "last_use_organism_step": self._last_use_organism_step.get(representation_ref),
                }
                for representation_ref in sorted(self._activation_counts)
            ],
            # `record()` evicts the oldest admitted representation when the
            # bounded tracker is full. Preserve that semantic insertion order
            # independently of the canonical ordering used for row encoding.
            "eviction_order": list(self._activation_counts),
            "capacity_pressure": self.pressure.checkpoint(),
            "sterile_reactivations": dict(sorted(self.sterile_reactivations.items())),
        }

    @classmethod
    def from_checkpoint(cls, payload: object) -> "GenerativeUseTracker":
        if not isinstance(payload, dict):
            raise ValueError("generative-use checkpoint must be an object")
        if set(payload) != {
            "schema_version",
            "representations",
            "eviction_order",
            "capacity_pressure",
            "sterile_reactivations",
        } or type(payload["schema_version"]) is not int or payload["schema_version"] != 2:
            raise ValueError("generative-use checkpoint fields are incomplete or unknown")
        raw = payload["representations"]
        if not isinstance(raw, list) or len(raw) > cls.MAX_REPRESENTATIONS:
            raise ValueError("invalid generative-use representation collection")
        raw_order = payload["eviction_order"]
        if not isinstance(raw_order, list) or len(raw_order) != len(raw):
            raise ValueError("invalid generative-use eviction order")
        if any(not isinstance(ref, str) for ref in raw_order):
            raise ValueError("generative-use eviction order entries must be strings")
        tracker = cls()
        seen_representations: set[str] = set()
        rows_by_ref: dict[str, dict[str, object]] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("invalid generative-use representation")
            if set(item) != {
                "representation_ref",
                "activation_count",
                "episode_ids",
                "source_refs",
                "hypothesis_count",
                "disagreement_sum",
                "disagreement_count",
                "last_use_organism_step",
            }:
                raise ValueError("generative-use representation fields are incomplete or unknown")
            representation_ref = bounded_identifier(
                item["representation_ref"], name="representation_ref"
            )
            if representation_ref in seen_representations:
                raise ValueError("duplicate generative-use representation")
            seen_representations.add(representation_ref)
            rows_by_ref[representation_ref] = item
        if len(set(raw_order)) != len(raw_order) or set(raw_order) != seen_representations:
            raise ValueError("generative-use eviction order must match representations exactly")
        for representation_ref in raw_order:
            item = rows_by_ref[representation_ref]
            activation_count = _checkpoint_int(item["activation_count"], "activation_count")
            hypothesis_count = _checkpoint_int(item["hypothesis_count"], "hypothesis_count")
            disagreement_sum = _checkpoint_number(item["disagreement_sum"], "disagreement_sum")
            disagreement_count = _checkpoint_int(item["disagreement_count"], "disagreement_count")
            last_use_organism_step = item.get("last_use_organism_step")
            if not isinstance(item["episode_ids"], list) or not isinstance(item["source_refs"], list):
                raise ValueError("generative-use episode_ids and source_refs must be lists")
            if any(not isinstance(value, str) for value in (*item["episode_ids"], *item["source_refs"])):
                raise ValueError("generative-use references must be strings")
            episode_ids = tuple(item["episode_ids"])
            source_refs = tuple(item["source_refs"])
            if len(set(episode_ids)) != len(episode_ids) or len(set(source_refs)) != len(source_refs):
                raise ValueError("generative-use references must be unique")
            if (
                activation_count < 0
                or hypothesis_count < 0
                or disagreement_sum < 0.0
                or disagreement_count < 0
                or (
                    last_use_organism_step is not None
                    and (
                        isinstance(last_use_organism_step, bool)
                        or not isinstance(last_use_organism_step, int)
                        or last_use_organism_step < 0
                    )
                )
                or len(episode_ids) > cls.MAX_EPISODES_PER_REPRESENTATION
                or len(source_refs) > cls.MAX_SOURCES_PER_REPRESENTATION
            ):
                raise ValueError("invalid generative-use aggregate")
            tracker._activation_counts[representation_ref] = activation_count
            tracker._episodes[representation_ref] = {
                bounded_identifier(value, name="episode_id") for value in episode_ids
            }
            tracker._sources[representation_ref] = {
                bounded_identifier(value, name="source_ref") for value in source_refs
            }
            tracker._hypothesis_counts[representation_ref] = hypothesis_count
            tracker._disagreement_sum[representation_ref] = disagreement_sum
            tracker._disagreement_count[representation_ref] = disagreement_count
            if last_use_organism_step is not None:
                tracker._last_use_organism_step[representation_ref] = last_use_organism_step
        tracker.pressure = CapacityPressure.restore(
            payload["capacity_pressure"], capacity=cls.MAX_REPRESENTATIONS
        )
        raw_sterile = payload["sterile_reactivations"]
        if not isinstance(raw_sterile, dict):
            raise ValueError("sterile_reactivations must be an object")
        tracker.sterile_reactivations = {}
        for ref, count in raw_sterile.items():
            bounded_identifier(ref, name="sterile reactivation representation_ref")
            if ref not in tracker._activation_counts:
                raise ValueError("sterile reactivation references unknown representation")
            tracker.sterile_reactivations[ref] = _checkpoint_int(count, "sterile reactivation count")
            if tracker.sterile_reactivations[ref] < 0:
                raise ValueError("sterile reactivation count must be non-negative")
        return tracker

    def _drop(self, representation_ref: str) -> None:
        self._activation_counts.pop(representation_ref, None)
        self._episodes.pop(representation_ref, None)
        self._sources.pop(representation_ref, None)
        self._hypothesis_counts.pop(representation_ref, None)
        self._disagreement_sum.pop(representation_ref, None)
        self._disagreement_count.pop(representation_ref, None)
        self._last_use_organism_step.pop(representation_ref, None)
        self.sterile_reactivations.pop(representation_ref, None)


class GenerativeConsolidator:
    """Produces signals only; structural admission remains another component's job."""

    def __init__(
        self,
        *,
        tracker: GenerativeUseTracker | None = None,
        minimum_cross_episode_reuse: int = 1,
        minimum_source_diversity: int = 2,
        minimum_demand: float = 0.5,
        retention_decay_steps: int = 32,
    ) -> None:
        self.tracker = tracker or GenerativeUseTracker()
        if (
            isinstance(minimum_cross_episode_reuse, bool)
            or not isinstance(minimum_cross_episode_reuse, int)
            or minimum_cross_episode_reuse < 0
        ):
            raise ValueError("minimum_cross_episode_reuse must be a non-negative integer")
        if (
            isinstance(minimum_source_diversity, bool)
            or not isinstance(minimum_source_diversity, int)
            or minimum_source_diversity < 0
        ):
            raise ValueError("minimum_source_diversity must be a non-negative integer")
        unit_interval(minimum_demand, name="minimum_demand")
        if (
            isinstance(retention_decay_steps, bool)
            or not isinstance(retention_decay_steps, int)
            or retention_decay_steps <= 0
        ):
            raise ValueError("retention_decay_steps must be a positive integer")
        self.minimum_cross_episode_reuse = minimum_cross_episode_reuse
        self.minimum_source_diversity = minimum_source_diversity
        self.minimum_demand = minimum_demand
        self.retention_decay_steps = retention_decay_steps

    def signal(
        self, *, representation_ref: str, generative_demand: float
    ) -> GenerativeConsolidationSignal:
        return self.tracker.signal(
            representation_ref=representation_ref, generative_demand=generative_demand
        )

    def demand_for_age(self, age_steps: int) -> float:
        if isinstance(age_steps, bool) or not isinstance(age_steps, int) or age_steps < 0:
            raise ValueError("age_steps must be a non-negative integer")
        return 1.0 / (1.0 + (age_steps / self.retention_decay_steps))

    def is_mature(self, signal: GenerativeConsolidationSignal) -> bool:
        return (
            signal.cross_episode_reuse >= self.minimum_cross_episode_reuse
            and signal.source_diversity >= self.minimum_source_diversity
            and signal.generative_demand >= self.minimum_demand
        )

    def project_candidate(
        self,
        *,
        signal: GenerativeConsolidationSignal,
        candidate_id: str,
        eligible_tick: int,
        mutation_payloads: tuple[str, ...],
    ) -> GenerativeCandidateProjection | None:
        """Create a contention request, never apply a graph mutation."""

        if not self.is_mature(signal):
            return None
        return GenerativeCandidateProjection(
            candidate_id=candidate_id,
            family="generative",
            producer_id="producer.generative",
            eligible_tick=eligible_tick,
            mutation_payloads=mutation_payloads,
        )

    def submit_candidate(
        self,
        projection: GenerativeCandidateProjection,
        *,
        register: Callable[..., bool],
        mutations: tuple[object, ...],
    ) -> bool:
        """Delegate admission to an external StructuralContention owner."""

        if not isinstance(mutations, tuple) or not mutations:
            raise ValueError("mutations must be a non-empty tuple")
        return bool(
            register(
                candidate_id=projection.candidate_id,
                family=projection.family,
                producer_id=projection.producer_id,
                eligible_tick=projection.eligible_tick,
                mutations=mutations,
            )
        )


__all__ = [
    "GenerativeCandidateProjection",
    "GenerativeConsolidationSignal",
    "GenerativeConsolidator",
    "GenerativeUseTracker",
]
