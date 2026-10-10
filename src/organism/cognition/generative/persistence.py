"""Fail-closed checkpoint encoding for a generative workspace."""

from __future__ import annotations

import json
from typing import Any

from .agenda import GenerativeAgenda
from .budget import GenerativeBudget
from .scheduler import GenerativeScheduler
from .types import (
    EpistemicOrigin,
    GeneratedFeature,
    GenerativeEpisode,
    GenerativeMode,
    GenerativeOperation,
    GenerativeState,
    GenerativeTermination,
    GenerativeTransition,
)
from .workspace import GenerativeWorkspace

GENERATIVE_COGNITION_SCHEMA_VERSION = 1


def dumps(workspace: GenerativeWorkspace) -> str:
    return json.dumps(
        {
            "schema_version": GENERATIVE_COGNITION_SCHEMA_VERSION,
            "workspace": workspace.checkpoint(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def loads(payload: str) -> GenerativeWorkspace:
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid generative checkpoint JSON") from exc
    return restore(decoded)


def dumps_cognition(
    *,
    workspace: GenerativeWorkspace,
    agenda: GenerativeAgenda,
    scheduler: GenerativeScheduler,
) -> str:
    """Encode workspace, agenda and scheduler as one fail-closed checkpoint."""
    return json.dumps(
        {
            "schema_version": GENERATIVE_COGNITION_SCHEMA_VERSION,
            "workspace": workspace.checkpoint(),
            "agenda": agenda.checkpoint(),
            "scheduler": scheduler.checkpoint(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def loads_cognition(
    payload: str,
) -> tuple[GenerativeWorkspace, GenerativeAgenda, GenerativeScheduler]:
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid generative cognition JSON") from exc
    return restore_cognition(decoded)


def restore(payload: object) -> GenerativeWorkspace:
    if not isinstance(payload, dict):
        raise ValueError("unsupported or missing generative cognition schema version")
    if "schema_version" not in payload or type(payload["schema_version"]) is not int or payload["schema_version"] != GENERATIVE_COGNITION_SCHEMA_VERSION:
        raise ValueError("unsupported or missing generative cognition schema version")
    if set(payload) != {"schema_version", "workspace"}:
        raise ValueError("invalid generative cognition checkpoint fields")
    raw_workspace = payload["workspace"]
    if not isinstance(raw_workspace, dict):
        raise ValueError("generative checkpoint workspace must be an object")
    expected_workspace_keys = {"episode", "budget", "states", "transitions", "model_queries", "closed"}
    if set(raw_workspace) != expected_workspace_keys:
        raise ValueError("invalid generative workspace checkpoint fields")
    episode = _episode(raw_workspace["episode"])
    expected_counts = (episode.state_count, episode.transition_count, episode.branch_count, episode.max_depth_reached)
    # Reconstruct counters from retained states/transitions instead of starting
    # from the persisted values and counting the same records a second time.
    episode.state_count = 0
    episode.transition_count = 0
    episode.branch_count = 0
    episode.max_depth_reached = 0
    budget = _budget(raw_workspace["budget"])
    workspace = GenerativeWorkspace(episode=episode, budget=budget)
    raw_states = raw_workspace["states"]
    raw_transitions = raw_workspace["transitions"]
    if not isinstance(raw_states, list) or not isinstance(raw_transitions, list):
        raise ValueError("generative checkpoint collections must be lists")
    for raw_state in raw_states:
        workspace.add_state(_state(raw_state))
    for raw_transition in raw_transitions:
        workspace.add_transition(_transition(raw_transition))
    queries = raw_workspace["model_queries"]
    if (
        isinstance(queries, bool)
        or not isinstance(queries, int)
        or not 0 <= queries <= budget.max_model_queries
    ):
        raise ValueError("invalid generative model query count")
    for _ in range(queries):
        workspace.consume_model_query()
    closed = raw_workspace["closed"]
    if not isinstance(closed, bool):
        raise ValueError("workspace closed flag must be boolean")
    if closed:
        if episode.termination_reason is None:
            raise ValueError("closed workspace must have a termination reason")
        workspace._closed = True  # checkpoint restore, not a public mutation path
    actual_counts = (episode.state_count, episode.transition_count, episode.branch_count, episode.max_depth_reached)
    if actual_counts != expected_counts:
        raise ValueError("generative episode counters do not match checkpoint contents")
    return workspace


def restore_cognition(
    payload: object,
) -> tuple[GenerativeWorkspace, GenerativeAgenda, GenerativeScheduler]:
    if not isinstance(payload, dict):
        raise ValueError("unsupported or missing generative cognition schema version")
    if "schema_version" not in payload or type(payload["schema_version"]) is not int or payload["schema_version"] != GENERATIVE_COGNITION_SCHEMA_VERSION:
        raise ValueError("unsupported or missing generative cognition schema version")
    if set(payload) != {"schema_version", "workspace", "agenda", "scheduler"}:
        raise ValueError("invalid generative cognition checkpoint fields")
    try:
        workspace = restore(
            {"schema_version": payload["schema_version"], "workspace": payload["workspace"]}
        )
        agenda = GenerativeAgenda.from_checkpoint(payload["agenda"])
        scheduler = GenerativeScheduler.from_checkpoint(payload["scheduler"])
        return workspace, agenda, scheduler
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid generative cognition checkpoint") from exc


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _counter(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _budget(value: Any) -> GenerativeBudget:
    raw = _object(value, "budget")
    names = {"max_states", "max_transitions", "max_depth", "max_branches", "max_model_queries"}
    if set(raw) != names:
        raise ValueError("invalid generative budget fields")
    try:
        return GenerativeBudget(
            **{
                name: raw[name]
                for name in (
                    "max_states",
                    "max_transitions",
                    "max_depth",
                    "max_branches",
                    "max_model_queries",
                )
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid generative budget") from exc


def _episode(value: Any) -> GenerativeEpisode:
    raw = _object(value, "episode")
    names = {"episode_id", "organism_id", "target_id", "root_state_id", "mode", "started_symbiont_tick", "started_generative_tick", "source_episode_ids", "state_count", "transition_count", "branch_count", "max_depth_reached", "termination_reason"}
    if set(raw) != names:
        raise ValueError("invalid generative episode fields")
    try:
        termination = raw["termination_reason"]
        source_ids = raw["source_episode_ids"]
        if not isinstance(source_ids, list):
            raise ValueError("source_episode_ids must be a list")
        return GenerativeEpisode(
            episode_id=raw["episode_id"],
            organism_id=raw["organism_id"],
            target_id=raw["target_id"],
            root_state_id=raw["root_state_id"],
            mode=GenerativeMode(raw["mode"]),
            started_symbiont_tick=raw["started_symbiont_tick"],
            started_generative_tick=raw["started_generative_tick"],
            source_episode_ids=tuple(source_ids),
            state_count=_counter(raw["state_count"], "state_count"),
            transition_count=_counter(raw["transition_count"], "transition_count"),
            branch_count=_counter(raw["branch_count"], "branch_count"),
            max_depth_reached=_counter(raw["max_depth_reached"], "max_depth_reached"),
            termination_reason=GenerativeTermination(termination)
            if termination is not None
            else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid generative episode") from exc


def _state(value: Any) -> GenerativeState:
    raw = _object(value, "state")
    names = {"state_id", "episode_id", "origin", "parent_state_id", "depth", "features", "active_concept_ids", "relation_refs", "source_episode_ids", "source_model_ids", "source_state_ids", "uncertainty", "coherence", "generative_tick"}
    if set(raw) != names:
        raise ValueError("invalid generative state fields")
    try:
        features_raw = raw["features"]
        if not isinstance(features_raw, list):
            raise ValueError("features must be a list")
        if any(not isinstance(item, dict) or set(item) != {"token", "value_class", "confidence", "source_model_id"} for item in features_raw):
            raise ValueError("invalid generated feature fields")
        features = tuple(GeneratedFeature(**item) for item in features_raw)
        collections = ("active_concept_ids", "relation_refs", "source_episode_ids", "source_model_ids", "source_state_ids")
        if any(not isinstance(raw[name], list) for name in collections):
            raise ValueError("generative state references must be lists")
        return GenerativeState(
            state_id=raw["state_id"],
            episode_id=raw["episode_id"],
            origin=EpistemicOrigin(raw["origin"]),
            parent_state_id=raw["parent_state_id"],
            depth=raw["depth"],
            features=features,
            active_concept_ids=tuple(raw["active_concept_ids"]),
            relation_refs=tuple(raw["relation_refs"]),
            source_episode_ids=tuple(raw["source_episode_ids"]),
            source_model_ids=tuple(raw["source_model_ids"]),
            source_state_ids=tuple(raw["source_state_ids"]),
            uncertainty=raw["uncertainty"],
            coherence=raw["coherence"],
            generative_tick=raw["generative_tick"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid generative state") from exc


def _transition(value: Any) -> GenerativeTransition:
    raw = _object(value, "transition")
    names = {"transition_id", "episode_id", "source_state_id", "target_state_id", "operation", "model_ids", "uncertainty_before", "uncertainty_after", "predicted_outcomes", "generative_tick"}
    if set(raw) != names:
        raise ValueError("invalid generative transition fields")
    try:
        if not isinstance(raw["model_ids"], list) or not isinstance(raw["predicted_outcomes"], list):
            raise ValueError("transition collections must be lists")
        return GenerativeTransition(
            transition_id=raw["transition_id"],
            episode_id=raw["episode_id"],
            source_state_id=raw["source_state_id"],
            target_state_id=raw["target_state_id"],
            operation=GenerativeOperation(raw["operation"]),
            model_ids=tuple(raw["model_ids"]),
            uncertainty_before=raw["uncertainty_before"],
            uncertainty_after=raw["uncertainty_after"],
            predicted_outcomes=tuple(raw["predicted_outcomes"]),
            generative_tick=raw["generative_tick"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid generative transition") from exc


__all__ = [
    "GENERATIVE_COGNITION_SCHEMA_VERSION",
    "dumps",
    "dumps_cognition",
    "loads",
    "loads_cognition",
    "restore",
    "restore_cognition",
]
