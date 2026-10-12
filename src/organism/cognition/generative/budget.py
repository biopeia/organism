"""Explicit engineering limits for one generative workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class GenerativeBudget:
    max_states: int = 64
    max_transitions: int = 64
    max_depth: int = 8
    max_branches: int = 8
    max_model_queries: int = 128

    def __post_init__(self) -> None:
        for name in (
            "max_states",
            "max_transitions",
            "max_depth",
            "max_branches",
            "max_model_queries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def checkpoint(self) -> dict[str, int]:
        return {
            "max_states": self.max_states,
            "max_transitions": self.max_transitions,
            "max_depth": self.max_depth,
            "max_branches": self.max_branches,
            "max_model_queries": self.max_model_queries,
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> GenerativeBudget:
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint payload must be a Mapping, got {type(payload).__name__}")
        expected_keys = {
            "max_states",
            "max_transitions",
            "max_depth",
            "max_branches",
            "max_model_queries",
        }
        if set(payload.keys()) != expected_keys:
            raise ValueError(
                f"checkpoint fields do not match expected schema: expected {sorted(expected_keys)}, got {sorted(payload.keys())}"
            )
        values: dict[str, int] = {}
        for name in sorted(expected_keys):
            val = payload[name]
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            values[name] = val
        return cls(**values)


__all__ = ["GenerativeBudget"]
