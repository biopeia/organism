import copy
import json

import pytest

from organism.cognition.generative import (
    GenerativeAgenda,
    GenerativeMode,
    GenerativeScheduler,
    dumps_cognition,
    loads_cognition,
    new_episode,
)
from organism.cognition.generative.workspace import GenerativeWorkspace
from organism.cognition.generative.types import (
    EpistemicOrigin,
    GenerativeOperation,
    GenerativeState,
    GenerativeTransition,
)


def test_cognition_checkpoint_round_trips_workspace_agenda_and_scheduler():
    workspace = GenerativeWorkspace(
        episode=new_episode(
            episode_id="e0",
            organism_id="o0",
            root_state_id="s0",
            mode=GenerativeMode.OFFLINE,
            symbiont_tick=4,
            generative_tick=2,
        )
    )
    agenda = GenerativeAgenda()
    scheduler = GenerativeScheduler(offline_quota=2)
    scheduler.schedule(
        mode=GenerativeMode.OFFLINE,
        tick=4,
        has_target=False,
        workspace_open=True,
        model_queries_remaining=1,
    )

    restored_workspace, restored_agenda, restored_scheduler = loads_cognition(
        dumps_cognition(workspace=workspace, agenda=agenda, scheduler=scheduler)
    )

    assert restored_workspace.checkpoint() == workspace.checkpoint()
    assert restored_agenda.checkpoint() == agenda.checkpoint()
    assert restored_scheduler.checkpoint() == scheduler.checkpoint()


def test_cognition_checkpoint_requires_all_domains():
    payload = {
        "schema_version": 1,
        "workspace": {},
        "agenda": {},
    }
    try:
        loads_cognition(__import__("json").dumps(payload))
    except ValueError as exc:
        assert "invalid generative cognition checkpoint" in str(exc)
    else:
        raise AssertionError("incomplete cognition checkpoint was accepted")
def test_workspace_restore_rejects_missing_or_unknown_emitted_fields():
    workspace = GenerativeWorkspace(
        episode=new_episode(
            episode_id="e1", organism_id="o1", root_state_id="s1",
            mode=GenerativeMode.ONLINE, symbiont_tick=1, generative_tick=0,
        )
    )
    checkpoint = json.loads(dumps_cognition(
        workspace=workspace, agenda=GenerativeAgenda(), scheduler=GenerativeScheduler()
    ))
    for field in ("schema_version", "workspace", "agenda", "scheduler"):
        missing = copy.deepcopy(checkpoint)
        del missing[field]
        with pytest.raises(ValueError):
            loads_cognition(json.dumps(missing))
    missing_state = copy.deepcopy(checkpoint)
    del missing_state["workspace"]["model_queries"]
    with pytest.raises(ValueError):
        loads_cognition(json.dumps(missing_state))
    extra = copy.deepcopy(checkpoint)
    extra["workspace"]["unexpected"] = 1
    with pytest.raises(ValueError):
        loads_cognition(json.dumps(extra))


def test_workspace_restore_rejects_episode_counter_mismatch():
    workspace = GenerativeWorkspace(
        episode=new_episode(
            episode_id="e2", organism_id="o2", root_state_id="s2",
            mode=GenerativeMode.ONLINE, symbiont_tick=1, generative_tick=0,
        )
    )
    payload = json.loads(dumps_cognition(
        workspace=workspace, agenda=GenerativeAgenda(), scheduler=GenerativeScheduler()
    ))
    payload["workspace"]["episode"]["state_count"] = 0
    payload["workspace"]["states"] = [
        {
            "state_id": "s2", "episode_id": "e2", "origin": "inferred",
            "parent_state_id": None, "depth": 0, "features": [],
            "active_concept_ids": [], "relation_refs": [], "source_episode_ids": [],
            "source_model_ids": [], "source_state_ids": [], "uncertainty": 0.1,
            "coherence": 1.0, "generative_tick": 0,
        }
    ]
    with pytest.raises(ValueError, match="invalid generative cognition checkpoint"):
        loads_cognition(json.dumps(payload))


def test_branch_transition_counter_round_trips_without_double_counting():
    workspace = GenerativeWorkspace(
        episode=new_episode(
            episode_id="branch-episode", organism_id="o", root_state_id="root",
            mode=GenerativeMode.OFFLINE, symbiont_tick=1, generative_tick=0,
        )
    )
    workspace.add_state(GenerativeState(
        state_id="root", episode_id="branch-episode", origin=EpistemicOrigin.OBSERVED,
        parent_state_id=None, depth=0, features=(), active_concept_ids=(), relation_refs=(),
        source_episode_ids=(), source_model_ids=(), source_state_ids=(), uncertainty=0.0,
        coherence=1.0, generative_tick=0,
    ))
    workspace.add_state(GenerativeState(
        state_id="branch", episode_id="branch-episode", origin=EpistemicOrigin.COUNTERFACTUAL,
        parent_state_id="root", depth=1, features=(), active_concept_ids=(), relation_refs=(),
        source_episode_ids=(), source_model_ids=(), source_state_ids=(), uncertainty=0.0,
        coherence=1.0, generative_tick=1,
    ))
    workspace.add_transition(GenerativeTransition(
        transition_id="branch-transition", episode_id="branch-episode",
        source_state_id="root", target_state_id="branch", operation=GenerativeOperation.BRANCH,
        model_ids=(), uncertainty_before=0.0, uncertainty_after=0.0,
        predicted_outcomes=(), generative_tick=1,
    ))

    restored = loads_cognition(dumps_cognition(
        workspace=workspace, agenda=GenerativeAgenda(), scheduler=GenerativeScheduler()
    ))[0]
    assert restored.checkpoint() == workspace.checkpoint()
