"""Exploration composer with the compact v5 checkpoint (organism#183, M3).

The v5 composer must behave exactly like the v4 composer: same candidates and
commands every tick, and its checkpoint must reconstruct the v4 bytes the v4
composer writes, across restores from committed checkpoints.
"""

from __future__ import annotations

import pytest

from organism.actuation import action_history as ah
from organism.actuation.bounded_action import (
    ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
    ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA,
    OrganismCheckpointContract,
    OrganismExplorationComposer,
)
from organism.actuation.domain import ActionDomain
from organism.actuation.surface import ActuatorSurface


from .test_exploration_composition import (
    ACTUATORS, IDENTITY, CompetenceDevelopmentEngine, _CommittedCut, _context, _transition,
)


def _engine(context):
    return CompetenceDevelopmentEngine(ACTUATORS, organism_id=str(IDENTITY), embodiment_fingerprint=context.fingerprint)


def _composers():
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)
    v4 = OrganismExplorationComposer(organism_id=str(IDENTITY), engine=_engine(context))
    v5 = OrganismExplorationComposer(organism_id=str(IDENTITY), engine=_engine(context), history=ah.HistoryIndex())
    return surface, context, v4, v5


def _prepare(composer, surface, context, tick):
    return composer.prepare(_transition(tick=tick), actuator_context=context, surface=surface,
                            target_causal_tick=tick + 1, physical_step_id=f"physical-step.{tick}",
                            commit_id=f"commit.{tick}")


def test_v5_composer_matches_v4_across_ticks_and_restores():
    surface, context, v4, v5 = _composers()
    store: dict[str, bytes] = {}
    for tick in range(1, 61):
        c4, c5 = _prepare(v4, surface, context, tick), _prepare(v5, surface, context, tick)
        assert (c5.proposal, c5.motor_command) == (c4.proposal, c4.motor_command)
        transition = _transition(tick=tick)
        b4 = v4.checkpoint_for_candidate(c4, transition_state=transition)
        b5, decl = v5.checkpoint_with_dependency_for_candidate(c5, transition_state=transition)
        if decl is not None and decl.new_blob is not None:
            store[decl.head] = decl.new_blob                     # World persists before commit
        assert ah.exploration_v5_to_v4(b5, resolve=store.__getitem__) == b4
        cut = _CommittedCut(c4.world_id, c4.target_causal_tick, c4.commit_id)
        v4.adopt_after_world_commit(c4, committed_cut=cut)
        v5.adopt_after_world_commit(c5, committed_cut=cut)
        assert len(v5._domain._commitments) <= 1                 # live records only in memory
        if tick % 10 == 0:                                       # fresh process from the committed cut
            v5 = OrganismExplorationComposer.restore_from_committed_checkpoint(
                b5, committed_cut=cut, expected_organism_id=str(IDENTITY),
                expected_physical_step_id=c5.physical_step_id, read_dependency=store.__getitem__,
            ).action_composer
            v4 = OrganismExplorationComposer.restore_from_committed_checkpoint(
                b4, committed_cut=cut, expected_organism_id=str(IDENTITY),
                expected_physical_step_id=c4.physical_step_id,
            ).action_composer


def test_uncommitted_v5_candidate_blocks_further_preparation():
    surface, context, _, v5 = _composers()
    candidate = _prepare(v5, surface, context, 1)
    v5.checkpoint_with_dependency_for_candidate(candidate, transition_state=_transition(tick=1))
    with pytest.raises(RuntimeError, match="uncommitted candidate"):
        _prepare(v5, surface, context, 2)


def test_v5_composer_refuses_dependency_free_export_and_resolver_free_restore():
    surface, context, _, v5 = _composers()
    candidate = _prepare(v5, surface, context, 1)
    with pytest.raises(ValueError, match="must export its dependency"):
        v5.checkpoint_for_candidate(candidate, transition_state=_transition(tick=1))
    encoded, _ = v5.checkpoint_with_dependency_for_candidate(candidate, transition_state=_transition(tick=1))
    with pytest.raises(ValueError, match="dependency resolver"):
        OrganismExplorationComposer.restore_from_committed_checkpoint(
            encoded, committed_cut=_CommittedCut(candidate.world_id, 2, candidate.commit_id),
            expected_organism_id=str(IDENTITY), expected_physical_step_id=candidate.physical_step_id)


def test_owner_contract_validates_v5_and_states_its_dependency():
    surface, context, v4, v5 = _composers()
    store: dict[str, bytes] = {}
    for tick in range(1, 4):                       # make something terminal so a head exists
        c5 = _prepare(v5, surface, context, tick)
        b5, decl = v5.checkpoint_with_dependency_for_candidate(c5, transition_state=_transition(tick=tick))
        if decl is not None and decl.new_blob is not None:
            store[decl.head] = decl.new_blob
        v5.adopt_after_world_commit(c5, committed_cut=_CommittedCut(c5.world_id, tick + 1, c5.commit_id))
    contract = OrganismCheckpointContract(read_dependency=store.__getitem__)
    assert contract.validate_checkpoint(b5, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
    assert contract.declared_dependency(b5, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA) == decl.head
    assert decl.head is not None
    # without the dependency the checkpoint is not valid
    assert not OrganismCheckpointContract().validate_checkpoint(b5, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
    c4 = _prepare(v4, surface, context, 1)
    b4 = v4.checkpoint_for_candidate(c4, transition_state=_transition(tick=1))
    assert contract.declared_dependency(b4, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA) is None


def test_clone_preserves_retired_identity():
    domain = ActionDomain.restore_live(ActionDomain().checkpoint(), retired_intent_ids=frozenset({"intent.x"}),
                                       retired_commitment_ids=frozenset({"c.x"}), retired_closed_ids=frozenset({"c.x"}))
    copy = domain.clone()
    assert "c.x" in copy._retired_commitment_ids and "intent.x" in copy._retired_intent_ids
    # a plain restore drops it, which is why staged copies must use clone()
    assert not ActionDomain.restore(domain.checkpoint())._retired_commitment_ids


def test_exports_dependencies_flags_v5_mode_only():
    _, _, v4, v5 = _composers()
    assert v4.exports_dependencies is False and v5.exports_dependencies is True


def _v5_lineage(ticks):
    surface, context, _, v5 = _composers()
    store: dict[str, bytes] = {}
    checkpoints = []
    for tick in range(1, ticks + 1):
        c5 = _prepare(v5, surface, context, tick)
        b5, decl = v5.checkpoint_with_dependency_for_candidate(c5, transition_state=_transition(tick=tick))
        if decl is not None and decl.new_blob is not None:
            store[decl.head] = decl.new_blob
        v5.adopt_after_world_commit(c5, committed_cut=_CommittedCut(c5.world_id, tick + 1, c5.commit_id))
        checkpoints.append((b5, decl))
    return store, checkpoints


def test_commit_validation_verifies_only_new_history_segments():
    store, checkpoints = _v5_lineage(40)
    reads: list[str] = []

    def resolve(digest):
        reads.append(digest)
        return store[digest]

    contract = OrganismCheckpointContract(read_dependency=resolve)
    per_validation = []
    for encoded, _ in checkpoints:
        before = len(reads)
        assert contract.validate_checkpoint(encoded, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
        per_validation.append(len(reads) - before)
    assert max(per_validation[1:]) <= 1            # constant per commit, not O(history)
    assert sum(per_validation) <= len(store)       # each segment read once in total
    # an older checkpoint (another branch point) is still valid: full cold verification
    assert contract.validate_checkpoint(checkpoints[5][0], str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)


def test_incremental_validation_refuses_corrupt_or_reused_identity():
    store, checkpoints = _v5_lineage(12)
    heads = [d.head for _, d in checkpoints if d is not None and d.new_blob is not None]
    contract = OrganismCheckpointContract(read_dependency=store.__getitem__)
    i = next(k for k, (_, d) in enumerate(checkpoints) if d is not None and d.new_blob is not None and d.head == heads[-1])
    for encoded, _ in checkpoints[:i]:
        contract.validate_checkpoint(encoded, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
    tampered = dict(store)
    tampered[heads[-1]] = store[heads[-1]].replace(b"exploration", b"exploratioN", 1)
    corrupt = OrganismCheckpointContract(read_dependency=tampered.__getitem__)
    corrupt._history_cache = contract._history_cache
    assert not corrupt.validate_checkpoint(checkpoints[i][0], str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
    # a validly-hashed segment that re-records an already retired id is refused
    import json as _json
    seg = _json.loads(store[heads[-1]])
    old = _json.loads(store[heads[0]])["records"][0]
    seg["records"] = seg["records"] + [old]
    forged = ah._canon(seg)
    forged_head = ah.sha256(forged).hexdigest()
    payload = _json.loads(checkpoints[i][0])
    payload["action_domain"]["history"] = {"head": forged_head, "count": payload["action_domain"]["history"]["count"] + 1}
    store2 = dict(store, **{forged_head: forged})
    forging = OrganismCheckpointContract(read_dependency=store2.__getitem__)
    forging._history_cache = contract._history_cache
    assert not forging.validate_checkpoint(ah._canon(payload), str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)


# ------------------------------------------ independent review regressions
def _forge(store, checkpoints, mutate_segment=None, mutate_payload=None):
    """Re-sign the last checkpoint's newest segment and/or payload after mutation."""
    import json as _json
    k = max(i for i, (_, d) in enumerate(checkpoints) if d is not None and d.new_blob is not None)
    payload = _json.loads(checkpoints[k][0])
    seg = _json.loads(store[checkpoints[k][1].head])
    store2 = dict(store)
    if mutate_segment:
        mutate_segment(seg)
        forged = ah._canon(seg)
        head = ah.sha256(forged).hexdigest()
        store2[head] = forged
        payload["action_domain"]["history"]["head"] = head
    if mutate_payload:
        mutate_payload(payload["action_domain"])
    return k, ah._canon(payload), store2


def _warm_and_cold(k, forged, store, checkpoints):
    warm = OrganismCheckpointContract(read_dependency=store.__getitem__)
    for encoded, _ in checkpoints[:k]:
        assert warm.validate_checkpoint(encoded, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
    cold = OrganismCheckpointContract(read_dependency=store.__getitem__)
    return (warm.validate_checkpoint(forged, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA),
            cold.validate_checkpoint(forged, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA))


@pytest.mark.parametrize("case", ["segment-seq-collision", "unknown-kind", "malformed-record"])
def test_incremental_and_cold_agree_on_forged_segments(case):
    store, checkpoints = _v5_lineage(20)

    def mutate(seg):
        rec = seg["records"][0]
        if case == "segment-seq-collision":
            rec["seq"] = 0                                 # seq already used by an older segment
        elif case == "unknown-kind":
            seg["records"].append({"kind": "bogus", "id": "c.future"})
        else:
            rec.pop("seq")

    k, forged, store2 = _forge(store, checkpoints, mutate_segment=mutate)
    assert _warm_and_cold(k, forged, store2, checkpoints) == (False, False)


@pytest.mark.parametrize("case", ["live-seq-collision", "rewound-counter"])
def test_ref_equal_warm_path_checks_live_seqs_and_counters(case):
    store, checkpoints = _v5_lineage(20)
    k = max(i for i, (_, d) in enumerate(checkpoints) if d is not None and d.new_blob is not None)

    def mutate(domain):
        if case == "live-seq-collision":
            domain["live_commitments"][0]["seq"] = 0
        else:
            domain["next_commitment_seq"] = 1
    import json as _json
    payload = _json.loads(checkpoints[k][0])
    mutate(payload["action_domain"])
    forged = ah._canon(payload)
    warm = OrganismCheckpointContract(read_dependency=store.__getitem__)
    for encoded, _ in checkpoints[: k + 1]:                 # cache now holds this exact ref
        warm.validate_checkpoint(encoded, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
    cold = OrganismCheckpointContract(read_dependency=store.__getitem__)
    assert not warm.validate_checkpoint(forged, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)
    assert not cold.validate_checkpoint(forged, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA)


def test_cached_index_never_crosses_owners():
    store, checkpoints = _v5_lineage(10)
    k = max(i for i, (_, d) in enumerate(checkpoints) if d is not None and d.new_blob is not None)
    import json as _json
    payload = _json.loads(checkpoints[k][0])["action_domain"]
    cache: dict = {}
    ah.restore_compact(payload, organism_id=str(IDENTITY), resolve=store.__getitem__, cache=cache)
    with pytest.raises(ah.ActionHistoryError):
        ah.restore_compact(payload, organism_id="organism.other", resolve=store.__getitem__, cache=cache)


def test_repeated_export_of_a_candidate_is_identical():
    surface, context, _, v5 = _composers()
    for tick in range(1, 4):
        candidate = _prepare(v5, surface, context, tick)
        first = v5.checkpoint_with_dependency_for_candidate(candidate, transition_state=_transition(tick=tick))
        assert v5.checkpoint_with_dependency_for_candidate(candidate, transition_state=_transition(tick=tick)) == first
        v5.adopt_after_world_commit(candidate, committed_cut=_CommittedCut(candidate.world_id, tick + 1,
                                                                           candidate.commit_id))


def test_v5_bytes_are_not_valid_under_the_v4_label():
    store, checkpoints = _v5_lineage(4)
    contract = OrganismCheckpointContract(read_dependency=store.__getitem__)
    assert not contract.validate_checkpoint(checkpoints[-1][0], str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA)


def test_composer_rejects_history_without_its_domain():
    store, checkpoints = _v5_lineage(10)
    import json as _json
    payload = _json.loads(checkpoints[-1][0])["action_domain"]
    _, index = ah.restore_compact(payload, organism_id=str(IDENTITY), resolve=store.__getitem__)
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    with pytest.raises(ValueError, match="requires the action domain"):
        OrganismExplorationComposer(organism_id=str(IDENTITY), engine=_engine(_context(surface)), history=index)
