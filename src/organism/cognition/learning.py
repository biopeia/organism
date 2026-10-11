from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Mapping

from .graph import CognitiveGraph, PlasticEdge
from .types import WEIGHT_RANGE, NodeKind

ELIGIBILITY_BOUND = (
    10.0  # generous relative to a_i, a_j in (-1, 1); prevents unbounded growth when decay==1.0
)

_VALID_SHADOW_STATUSES = frozenset({"candidate", "supported", "contradicted", "retired"})


def _require_finite_number(val: Any, field_name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {type(val).__name__}")
    f_val = float(val)
    if not math.isfinite(f_val):
        raise ValueError(f"{field_name} must be finite, got {f_val}")
    return f_val


def _require_non_negative_int(val: Any, field_name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{field_name} must be an integer, got {type(val).__name__}")
    if val < 0:
        raise ValueError(f"{field_name} must be non-negative, got {val}")
    return val


def _require_non_empty_str(val: Any, field_name: str) -> str:
    if not isinstance(val, str) or not val:
        raise ValueError(f"{field_name} must be a non-empty string")
    return val


def _require_status(val: Any) -> str:
    if not isinstance(val, str) or val not in _VALID_SHADOW_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(_VALID_SHADOW_STATUSES)}, got {val!r}"
        )
    return val


def huber_loss(error: float, delta: float = 1.0) -> float:
    magnitude = abs(error)
    if magnitude <= delta:
        return 0.5 * error * error
    return delta * (magnitude - 0.5 * delta)


@dataclass(slots=True, frozen=True)
class PredictionError:
    predictor_id: str
    target_id: str
    error: float
    loss: float


def compute_prediction_errors(
    graph: CognitiveGraph, *, current: Mapping[str, float], previous: Mapping[str, float]
) -> tuple[PredictionError, ...]:
    predictors = sorted(
        (node for node in graph.nodes if node.kind is NodeKind.PREDICTOR),
        key=lambda node: node.node_id,
    )
    errors: list[PredictionError] = []
    for predictor in predictors:
        if predictor.node_id not in previous:
            continue  # cold start: no prediction was made last tick
        target_id = predictor.predicts_node_id
        assert target_id is not None  # guaranteed by CognitiveGraph construction validation
        target_value = current.get(target_id, 0.0)
        predicted_value = previous[predictor.node_id]
        error = target_value - predicted_value
        errors.append(
            PredictionError(
                predictor_id=predictor.node_id,
                target_id=target_id,
                error=error,
                loss=huber_loss(error),
            )
        )
    return tuple(errors)


def update_eligibility(
    edge: PlasticEdge, *, source_previous: float, target_current: float, decay: float
) -> None:
    updated = decay * edge.eligibility + source_previous * target_current
    edge.eligibility = max(-ELIGIBILITY_BOUND, min(ELIGIBILITY_BOUND, updated))


def apply_oja_update(
    edge: PlasticEdge,
    *,
    source_activation: float,
    target_activation: float,
    learning_rate: float,
    modulation: float,
    eligible: bool,
    frozen: bool = False,
) -> None:
    if frozen or not eligible or modulation == 0.0:
        return
    delta = (
        learning_rate
        * modulation
        * (
            source_activation * target_activation
            - target_activation * target_activation * edge.weight
        )
    )
    new_weight = edge.weight + delta
    edge.weight = max(WEIGHT_RANGE[0], min(WEIGHT_RANGE[1], new_weight))


@dataclass(slots=True)
class ShadowPrediction:
    """Out-of-sample predictor candidate; never mutates the cognitive graph."""

    source_id: str
    target_id: str
    samples: int = 0
    model_loss: float = 0.0
    persistence_loss: float = 0.0
    status: str = "candidate"

    def __post_init__(self) -> None:
        self.source_id = _require_non_empty_str(self.source_id, "source_id")
        self.target_id = _require_non_empty_str(self.target_id, "target_id")
        self.samples = _require_non_negative_int(self.samples, "samples")
        self.model_loss = _require_finite_number(self.model_loss, "model_loss")
        self.persistence_loss = _require_finite_number(self.persistence_loss, "persistence_loss")
        if self.model_loss < 0.0:
            raise ValueError(f"model_loss must be non-negative, got {self.model_loss}")
        if self.persistence_loss < 0.0:
            raise ValueError(f"persistence_loss must be non-negative, got {self.persistence_loss}")
        self.status = _require_status(self.status)

    def observe(
        self, source_previous: float, target_current: float, target_previous: float
    ) -> None:
        # The source value is the one-step model prediction in shadow mode.
        self.samples += 1
        self.model_loss += huber_loss(target_current - source_previous)
        self.persistence_loss += huber_loss(target_current - target_previous)
        if self.samples >= 8 and self.status != "retired":
            self.status = "supported" if self.predictive_gain > 0.0 else "contradicted"
            if self.samples >= 16 and self.status == "contradicted":
                self.status = "retired"

    @property
    def predictive_gain(self) -> float:
        if self.samples == 0:
            return 0.0
        return (self.persistence_loss - self.model_loss) / self.samples

    @property
    def promotable(self) -> bool:
        return self.status == "supported" and self.samples >= 8 and self.predictive_gain > 0.0

    def checkpoint(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "samples": self.samples,
            "model_loss": self.model_loss,
            "persistence_loss": self.persistence_loss,
            "status": self.status,
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> ShadowPrediction:
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint payload must be a Mapping, got {type(payload).__name__}")
        expected_keys = {"source_id", "target_id", "samples", "model_loss", "persistence_loss", "status"}
        if set(payload.keys()) != expected_keys:
            raise ValueError(
                f"checkpoint fields do not match expected schema: expected {sorted(expected_keys)}, got {sorted(payload.keys())}"
            )
        return cls(
            source_id=_require_non_empty_str(payload["source_id"], "source_id"),
            target_id=_require_non_empty_str(payload["target_id"], "target_id"),
            samples=_require_non_negative_int(payload["samples"], "samples"),
            model_loss=_require_finite_number(payload["model_loss"], "model_loss"),
            persistence_loss=_require_finite_number(payload["persistence_loss"], "persistence_loss"),
            status=_require_status(payload["status"]),
        )



@dataclass(slots=True)
class LaggedShadowPrediction:
    """Bounded out-of-graph candidate for a fixed multi-step relation.

    This is deliberately not a causal claim and never mutates a
    ``CognitiveGraph``.  It only asks whether a source value observed a fixed
    number of ticks ago predicts a target better than target persistence.
    Keeping this experiment in shadow mode lets us measure temporal memory
    without silently expanding the canonical graph's one-step semantics.
    """

    source_id: str
    target_id: str
    lag_ticks: int
    samples: int = 0
    model_loss: float = 0.0
    persistence_loss: float = 0.0
    status: str = "candidate"
    _source_history: Deque[float] = field(init=False, repr=False)
    _target_history: Deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.source_id = _require_non_empty_str(self.source_id, "source_id")
        self.target_id = _require_non_empty_str(self.target_id, "target_id")
        if isinstance(self.lag_ticks, bool) or not isinstance(self.lag_ticks, int):
            raise ValueError("lag_ticks must be an integer")
        if not 1 <= self.lag_ticks <= 16:
            raise ValueError("lag_ticks must be within [1, 16]")
        self.samples = _require_non_negative_int(self.samples, "samples")
        self.model_loss = _require_finite_number(self.model_loss, "model_loss")
        self.persistence_loss = _require_finite_number(self.persistence_loss, "persistence_loss")
        if self.model_loss < 0.0:
            raise ValueError(f"model_loss must be non-negative, got {self.model_loss}")
        if self.persistence_loss < 0.0:
            raise ValueError(f"persistence_loss must be non-negative, got {self.persistence_loss}")
        self.status = _require_status(self.status)
        self._source_history: Deque[float] = deque(maxlen=self.lag_ticks + 1)
        self._target_history: Deque[float] = deque(maxlen=2)

    def observe(self, source_current: float, target_current: float) -> None:
        self._source_history.append(float(source_current))
        self._target_history.append(float(target_current))
        if len(self._source_history) <= self.lag_ticks or len(self._target_history) < 2:
            return

        source_value = self._source_history[0]
        target_previous = self._target_history[-2]
        self.samples += 1
        self.model_loss += huber_loss(target_current - source_value)
        self.persistence_loss += huber_loss(target_current - target_previous)
        if self.samples >= 8 and self.status != "retired":
            self.status = "supported" if self.predictive_gain > 0.0 else "contradicted"
            if self.samples >= 16 and self.status == "contradicted":
                self.status = "retired"

    @property
    def predictive_gain(self) -> float:
        if self.samples == 0:
            return 0.0
        return (self.persistence_loss - self.model_loss) / self.samples

    @property
    def promotable(self) -> bool:
        return self.status == "supported" and self.samples >= 8 and self.predictive_gain > 0.0

    def checkpoint(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "lag_ticks": self.lag_ticks,
            "samples": self.samples,
            "model_loss": self.model_loss,
            "persistence_loss": self.persistence_loss,
            "status": self.status,
            "source_history": list(self._source_history),
            "target_history": list(self._target_history),
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> LaggedShadowPrediction:
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint payload must be a Mapping, got {type(payload).__name__}")
        expected_keys = {
            "source_id",
            "target_id",
            "lag_ticks",
            "samples",
            "model_loss",
            "persistence_loss",
            "status",
            "source_history",
            "target_history",
        }
        if set(payload.keys()) != expected_keys:
            raise ValueError(
                f"checkpoint fields do not match expected schema: expected {sorted(expected_keys)}, got {sorted(payload.keys())}"
            )
        instance = cls(
            source_id=_require_non_empty_str(payload["source_id"], "source_id"),
            target_id=_require_non_empty_str(payload["target_id"], "target_id"),
            lag_ticks=payload["lag_ticks"],
            samples=_require_non_negative_int(payload["samples"], "samples"),
            model_loss=_require_finite_number(payload["model_loss"], "model_loss"),
            persistence_loss=_require_finite_number(payload["persistence_loss"], "persistence_loss"),
            status=_require_status(payload["status"]),
        )
        source_hist = payload["source_history"]
        target_hist = payload["target_history"]
        if not isinstance(source_hist, list) or not isinstance(target_hist, list):
            raise ValueError("source_history and target_history must be lists")
        if len(source_hist) > instance.lag_ticks + 1:
            raise ValueError("source_history exceeds maximum capacity for lag_ticks")
        if len(target_hist) > 2:
            raise ValueError("target_history exceeds maximum capacity")
        for val in source_hist:
            instance._source_history.append(_require_finite_number(val, "source_history item"))
        for val in target_hist:
            instance._target_history.append(_require_finite_number(val, "target_history item"))
        return instance


@dataclass(slots=True)
class ComposedShadowPrediction:
    """Experimental composition of two learned one-step relations.

    The candidate is deliberately limited to a three-channel chain.  It
    estimates ``m[t] ~= a*x[t-1]`` and ``y[t] ~= b*m[t-1]`` from observations,
    then evaluates the composed prediction ``a*b*x[t-2]`` against target
    persistence.  Rolling evidence makes the second relation revisable after
    a change in dynamics.  The candidate remains outside the graph.
    """

    source_id: str
    intermediate_id: str
    target_id: str
    window_ticks: int = 32
    samples: int = 0
    model_loss: float = 0.0
    persistence_loss: float = 0.0
    status: str = "candidate"
    _source_history: Deque[float] = field(init=False, repr=False)
    _intermediate_history: Deque[float] = field(init=False, repr=False)
    _target_history: Deque[float] = field(init=False, repr=False)
    _first_pairs: Deque[tuple[float, float]] = field(init=False, repr=False)
    _second_pairs: Deque[tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.source_id = _require_non_empty_str(self.source_id, "source_id")
        self.intermediate_id = _require_non_empty_str(self.intermediate_id, "intermediate_id")
        self.target_id = _require_non_empty_str(self.target_id, "target_id")
        if isinstance(self.window_ticks, bool) or not isinstance(self.window_ticks, int):
            raise ValueError("window_ticks must be an integer")
        if not 8 <= self.window_ticks <= 256:
            raise ValueError("window_ticks must be within [8, 256]")
        self.samples = _require_non_negative_int(self.samples, "samples")
        self.model_loss = _require_finite_number(self.model_loss, "model_loss")
        self.persistence_loss = _require_finite_number(self.persistence_loss, "persistence_loss")
        if self.model_loss < 0.0:
            raise ValueError(f"model_loss must be non-negative, got {self.model_loss}")
        if self.persistence_loss < 0.0:
            raise ValueError(f"persistence_loss must be non-negative, got {self.persistence_loss}")
        self.status = _require_status(self.status)
        self._source_history = deque(maxlen=3)
        self._intermediate_history = deque(maxlen=2)
        self._target_history = deque(maxlen=2)
        self._first_pairs = deque(maxlen=self.window_ticks)
        self._second_pairs = deque(maxlen=self.window_ticks)

    @staticmethod
    def _slope(pairs: Deque[tuple[float, float]]) -> float:
        denominator = sum(source * source for source, _ in pairs)
        if denominator <= 1e-12:
            return 0.0
        return sum(source * target for source, target in pairs) / denominator

    @property
    def first_relation_slope(self) -> float:
        return self._slope(self._first_pairs)

    @property
    def second_relation_slope(self) -> float:
        return self._slope(self._second_pairs)

    def observe(
        self, source_current: float, intermediate_current: float, target_current: float
    ) -> None:
        source_current = float(source_current)
        intermediate_current = float(intermediate_current)
        target_current = float(target_current)
        if self._source_history:
            self._first_pairs.append((self._source_history[-1], intermediate_current))
        if self._intermediate_history:
            self._second_pairs.append((self._intermediate_history[-1], target_current))

        self._source_history.append(source_current)
        self._intermediate_history.append(intermediate_current)
        self._target_history.append(target_current)
        if len(self._source_history) < 3 or len(self._target_history) < 2:
            return

        predicted = self.first_relation_slope * self.second_relation_slope * self._source_history[0]
        target_previous = self._target_history[-2]
        self.samples += 1
        self.model_loss += huber_loss(target_current - predicted)
        self.persistence_loss += huber_loss(target_current - target_previous)
        if self.samples >= 8 and self.status != "retired":
            self.status = "supported" if self.predictive_gain > 0.0 else "contradicted"
            if self.samples >= 16 and self.status == "contradicted":
                self.status = "retired"

    @property
    def predictive_gain(self) -> float:
        if self.samples == 0:
            return 0.0
        return (self.persistence_loss - self.model_loss) / self.samples

    @property
    def promotable(self) -> bool:
        return self.status == "supported" and self.samples >= 8 and self.predictive_gain > 0.0

    def checkpoint(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "intermediate_id": self.intermediate_id,
            "target_id": self.target_id,
            "window_ticks": self.window_ticks,
            "samples": self.samples,
            "model_loss": self.model_loss,
            "persistence_loss": self.persistence_loss,
            "status": self.status,
            "source_history": list(self._source_history),
            "intermediate_history": list(self._intermediate_history),
            "target_history": list(self._target_history),
            "first_pairs": [list(p) for p in self._first_pairs],
            "second_pairs": [list(p) for p in self._second_pairs],
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> ComposedShadowPrediction:
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint payload must be a Mapping, got {type(payload).__name__}")
        expected_keys = {
            "source_id",
            "intermediate_id",
            "target_id",
            "window_ticks",
            "samples",
            "model_loss",
            "persistence_loss",
            "status",
            "source_history",
            "intermediate_history",
            "target_history",
            "first_pairs",
            "second_pairs",
        }
        if set(payload.keys()) != expected_keys:
            raise ValueError(
                f"checkpoint fields do not match expected schema: expected {sorted(expected_keys)}, got {sorted(payload.keys())}"
            )
        instance = cls(
            source_id=_require_non_empty_str(payload["source_id"], "source_id"),
            intermediate_id=_require_non_empty_str(payload["intermediate_id"], "intermediate_id"),
            target_id=_require_non_empty_str(payload["target_id"], "target_id"),
            window_ticks=payload["window_ticks"],
            samples=_require_non_negative_int(payload["samples"], "samples"),
            model_loss=_require_finite_number(payload["model_loss"], "model_loss"),
            persistence_loss=_require_finite_number(payload["persistence_loss"], "persistence_loss"),
            status=_require_status(payload["status"]),
        )
        source_hist = payload["source_history"]
        inter_hist = payload["intermediate_history"]
        target_hist = payload["target_history"]
        first_pairs = payload["first_pairs"]
        second_pairs = payload["second_pairs"]
        for name, lst, max_len in (
            ("source_history", source_hist, 3),
            ("intermediate_history", inter_hist, 2),
            ("target_history", target_hist, 2),
        ):
            if not isinstance(lst, list):
                raise ValueError(f"{name} must be a list")
            if len(lst) > max_len:
                raise ValueError(f"{name} exceeds maximum capacity")

        for val in source_hist:
            instance._source_history.append(_require_finite_number(val, "source_history item"))
        for val in inter_hist:
            instance._intermediate_history.append(_require_finite_number(val, "intermediate_history item"))
        for val in target_hist:
            instance._target_history.append(_require_finite_number(val, "target_history item"))

        for name, pairs in (("first_pairs", first_pairs), ("second_pairs", second_pairs)):
            if not isinstance(pairs, list):
                raise ValueError(f"{name} must be a list")
            if len(pairs) > instance.window_ticks:
                raise ValueError(f"{name} exceeds window_ticks capacity")
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"{name} items must be 2-element tuples/lists")
                x = _require_finite_number(pair[0], f"{name} item[0]")
                y = _require_finite_number(pair[1], f"{name} item[1]")
                getattr(instance, f"_{name}").append((x, y))

        return instance
