"""organism#183 M3 prototype: equivalence with the full-history ActionDomain.

The oracle is the current ``ActionDomain`` round-tripped through its own
checkpoint after every operation (as bounded exploration does every step).
The compact form must give the same verdict, the same exception message and
the same live state for every operation, on both the warm (in-process index)
and the cold (history reload) restore paths.
"""

from __future__ import annotations

import copy
import json
import random
from hashlib import sha256

import pytest

from organism.actuation import action_history as ah
from organism.actuation.action import ActionEvaluation, ActionJustification, ActionProposal, ActionSource
from organism.actuation.controller import SequenceController
from organism.actuation.domain import ActionDomain
from organism.actuation.surface import derive_actuator_constitution
from organism.actuation.types import MotorIntent
from organism.agency.affordance import ActionAffordance
from organism.agency.executive_outcome import ExecutiveOutcomeLedger, ExecutiveOutcomeSample, OutcomeClass
from organism.agency.intention import IntentStatus
from organism.agency.prospective import ProspectiveDecision

ORG = "organism.test"
SURFACE = derive_actuator_constitution(("effector.left",))
TERMINAL_INTENT = {"satisfied", "failed", "rejected", "interrupted", "invalidated"}


# ------------------------------------------------------------------ fixtures
def _decision(n):
    return ProspectiveDecision(
        competence_id="competence.walk", anticipated_effect_id="effect.displacement",
        prediction_ref=f"prediction.{n}", confidence=0.5, epistemic_relevance=0.4,
        homeostatic_relevance=0.2, origin_refs=("source.readout",), supporting_affordance_id=f"affordance.{n}")


def _affordance(n):
    return ActionAffordance(
        affordance_id=f"affordance.{n}", competence_id="competence.walk",
        anticipated_effect_id="effect.displacement", context_ref="context.1", embodiment_id="embodiment.1",
        prediction_confidence=0.5, controllability=0.4, executability_confidence=1.0,
        prediction_ref=f"prediction.{n}", evidence_refs=("evidence.1",))


def apply(domain, op):
    kind, a = op
    if kind == "admit":
        domain.admit(_decision(a["n"]), supporting_affordance=_affordance(a["n"]), intent_id=a["id"],
                     context_ref="context.1", embodiment_id="embodiment.1", tick=a["tick"])
        return None
    if kind == "commit":
        if a.get("intent") is None:
            proposal = ActionProposal(proposal_id=f"p.{a['cid']}", source=ActionSource.EXPLORATION,
                                      effect_target_id=None, competence_id=None,
                                      justification=ActionJustification(), evaluation=ActionEvaluation())
            maximum = a.get("maxd", 1)
        else:
            proposal = ActionProposal(
                proposal_id=f"p.{a['cid']}", source=ActionSource.COMPETENCE,
                effect_target_id="effect.displacement", competence_id="competence.walk",
                justification=ActionJustification(effect_target_id="effect.displacement",
                                                  competence_id="competence.walk"),
                evaluation=ActionEvaluation(effect_confidence=a.get("conf", 0.5), uncertainty=0.2),
                intent_id=a["intent"])
            maximum = None
        selection = domain.select_and_commit(
            (proposal,), tick=a["tick"], commitment_id=a["cid"], controller_id=f"ctl.{a['cid']}",
            surface=SURFACE, embodiment_id="embodiment.1", interruptibility=a.get("intr", 1.0),
            minimum_duration=0, maximum_duration=maximum)
        return selection.commitment.commitment_id if selection.commitment else None
    if kind == "command":
        active = domain.active_commitment
        controller = SequenceController(controller_id=active.controller_id if active else "none",
                                        sequence=((MotorIntent("effector.left", a.get("v", 0.25)),),))
        command = domain.issue_command(controller, tick=a["tick"], command_id=a["id"])
        return (command.command_id, command.channels, command.commitment_id)
    if kind == "close":
        outcome = None
        if a.get("outcome"):
            outcome = ExecutiveOutcomeSample(tick=a["tick"], outcome_class=OutcomeClass.POSITIVE,
                                             reason=a["reason"], effect_similarity=1.0,
                                             prediction_mismatch=0.0, progress_before_failure=None)
        domain.close(a["id"], status=IntentStatus(a["status"]), tick=a["tick"], reason=a["reason"],
                     outcome=outcome)
        return None
    if kind == "close_exploration":
        domain.close_exploration(a["id"], tick=a["tick"], reason="probe_close")
        return None
    raise AssertionError(kind)


def verdict(domain, op):
    try:
        return ("ok", apply(domain, op), None)
    except Exception as exc:  # noqa: BLE001 - compare every rejection
        return ("err", type(exc).__name__, str(exc))


def live_state(domain):
    full = domain.checkpoint()
    return json.dumps({
        "commitments": [c for c in full["commitments"] if c["status"] == "active"],
        "intents": [i for i in full["intents"] if i["status"] not in TERMINAL_INTENT],
        "active": full["active_commitment_id"],
        "ledger": full["outcome_ledger"],
    }, sort_keys=True)


class Lockstep:
    """Oracle (full history) and M3 (compact + external history) side by side."""

    def __init__(self, base=None):
        base = base or ActionDomain(outcome_ledger=ExecutiveOutcomeLedger()).checkpoint()
        self.store: dict[str, bytes] = {}
        self.cache: dict = {}
        payload, segments = ah.v4_to_compact(base, organism_id=ORG)
        for seg in segments:
            self.store[sha256(seg).hexdigest()] = seg
        self.oracle_ckpts = [copy.deepcopy(base)]
        self.compact_ckpts = [payload]
        self.oracle = ActionDomain.restore(copy.deepcopy(base))
        self.domain, self.index = ah.restore_compact(payload, organism_id=ORG, resolve=self.store.__getitem__)

    def step(self, op, *, cold=False):
        expected, actual = verdict(self.oracle, op), verdict(self.domain, op)
        full = self.oracle.checkpoint()
        payload, segment, self.index = ah.checkpoint_compact(self.domain, self.index, organism_id=ORG)
        if segment is not None:
            self.store[sha256(segment).hexdigest()] = segment   # dependency durable before payload
        self.oracle_ckpts.append(full)
        self.compact_ckpts.append(payload)
        self.oracle = ActionDomain.restore(copy.deepcopy(full))
        self.domain, self.index = ah.restore_compact(
            payload, organism_id=ORG, resolve=self.store.__getitem__, cache=None if cold else self.cache)
        assert actual == expected, op
        assert live_state(self.domain) == live_state(self.oracle), op
        # the compact form always reconstructs the oracle's full section exactly
        assert ah.compact_to_v4(payload, organism_id=ORG, resolve=self.store.__getitem__) == full
        return expected

    def restore_old(self, back):
        k = max(1, min(back, len(self.oracle_ckpts)))
        self.oracle = ActionDomain.restore(copy.deepcopy(self.oracle_ckpts[-k]))
        self.domain, self.index = ah.restore_compact(
            self.compact_ckpts[-k], organism_id=ORG, resolve=self.store.__getitem__, cache=self.cache)
        self.oracle_ckpts.append(self.oracle_ckpts[-k])
        self.compact_ckpts.append(self.compact_ckpts[-k])


def _commit(cid, tick, intent=None, **kw):
    return ("commit", dict({"cid": cid, "tick": tick, "intent": intent}, **kw))


# ------------------------------------------------------ fixed regressions
# Each sequence is a minimal counterexample found against digest-only or
# watermark designs (M2a/M2b). M3 must behave exactly like full history.
REGRESSIONS = {
    "commitment id reused after close": [
        _commit("c1", 919), ("close_exploration", {"id": "c1", "tick": 944}), _commit("c1", 949)],
    "commitment id reused after replacement": [
        _commit("c25", 33), _commit("c36", 34), _commit("c25", 35)],
    "intent id reused after satisfaction": [
        ("admit", {"id": "intent.1", "n": 1, "tick": 1}), _commit("c1", 1, intent="intent.1"),
        ("close", {"id": "c1", "status": "satisfied", "reason": "anticipated_effect_observed", "tick": 2,
                   "outcome": True}),
        ("admit", {"id": "intent.1", "n": 2, "tick": 3})],
    "backward tick after close is legal": [
        _commit("c9", 15), ("close_exploration", {"id": "c9", "tick": 17}), _commit("c14", 14)],
    "several commitments in one tick": [
        _commit("a", 5), ("close_exploration", {"id": "a", "tick": 5}), _commit("b", 5),
        ("close_exploration", {"id": "b", "tick": 5}), _commit("a", 5)],
    "double close and close of a replaced commitment": [
        _commit("e1", 1, maxd=1), _commit("e2", 2, maxd=1),       # e1 expired: replaced, INTERRUPTED
        ("close_exploration", {"id": "e1", "tick": 3}),           # replaced is not in the closed set
        ("close_exploration", {"id": "e2", "tick": 3}),
        ("close_exploration", {"id": "e2", "tick": 4}),           # double close
        _commit("e1", 5)],                                        # replaced id still not reusable
    "commitment for a terminal intent": [
        ("admit", {"id": "intent.1", "n": 1, "tick": 1}), _commit("c1", 1, intent="intent.1"),
        ("close", {"id": "c1", "status": "invalidated", "reason": "context_lost", "tick": 2}),
        _commit("c2", 3, intent="intent.1")],
}


# Expected oracle verdicts, so a fixture that silently fails early cannot pass.
EXPECTED = {
    "commitment id reused after close": ["ok", "ok", "err"],
    "commitment id reused after replacement": ["ok", "ok", "err"],
    "intent id reused after satisfaction": ["ok", "ok", "ok", "err"],
    "backward tick after close is legal": ["ok", "ok", "ok"],
    "several commitments in one tick": ["ok", "ok", "ok", "ok", "err"],
    "double close and close of a replaced commitment": ["ok", "ok", "err", "ok", "err", "err"],
    "commitment for a terminal intent": ["ok", "ok", "ok", "err"],
}


@pytest.mark.parametrize("cold", [False, True], ids=["warm-index", "cold-reload"])
@pytest.mark.parametrize("name", sorted(REGRESSIONS))
def test_regression_matches_full_history(name, cold):
    run = Lockstep()
    verdicts = [run.step(op, cold=cold)[0] for op in REGRESSIONS[name]]
    assert verdicts == EXPECTED[name]


def test_old_restore_then_id_from_abandoned_future_matches_full_history():
    run = Lockstep()
    run.step(_commit("c1", 1))
    run.step(("close_exploration", {"id": "c1", "tick": 2}))
    # checkpoints: [genesis, c1 active, c1 closed]
    run.restore_old(3)                             # genesis: c1 never existed here
    assert run.step(_commit("c1", 3))[0] == "ok"   # full history agrees: id is fresh on this branch
    # checkpoints: [..., c1 closed, genesis, c1 active (new branch)]
    run.restore_old(3)                             # back to the branch where c1 was closed
    assert run.step(_commit("c1", 5))[0] == "err"  # ...where the id is already used


# ------------------------------------------------- deterministic differential
def _ops(seed, n):
    r = random.Random(seed)
    tick, counter, intents, commits = 1, 0, [], []
    for _ in range(n):
        if r.random() < 0.08:
            tick = max(0, tick - r.randint(1, 4))
        elif r.random() < 0.6:
            tick += r.choice((1, 1, 2))
        counter += 1
        roll = r.random()
        if roll < 0.04:
            yield ("restore_old", {"back": r.randint(1, 20)})
        elif roll < 0.22:
            iid = r.choice(intents) if intents and r.random() < 0.15 else f"intent.{counter}"
            intents.append(iid)
            yield ("admit", {"id": iid, "n": counter, "tick": tick})
        elif roll < 0.52:
            cid = r.choice(commits) if commits and r.random() < 0.12 else f"commit.{counter}"
            commits.append(cid)
            intent = None if r.random() < 0.45 or not intents else r.choice(intents[-6:])
            yield _commit(cid, tick, intent=intent, maxd=r.randint(1, 3), conf=r.choice((0.2, 0.5, 0.9)),
                          intr=r.choice((0.0, 0.5, 1.0)))
        elif roll < 0.66:
            yield ("command", {"id": f"cmd.{counter}", "tick": tick, "v": r.choice((0.0, 0.25, 0.75))})
        else:
            target = r.choice(commits) if commits else "commit.none"
            if r.random() < 0.5:
                status, reason = r.choice((("satisfied", "anticipated_effect_observed"),
                                           ("failed", "controller_failed"), ("invalidated", "context_lost"),
                                           ("interrupted", "operator_interrupt")))
                yield ("close", {"id": target, "status": status, "reason": reason, "tick": tick,
                                 "outcome": status == "satisfied" and r.random() < 0.9})
            else:
                yield ("close_exploration", {"id": target, "tick": tick})


@pytest.mark.parametrize("seed", range(12))
def test_seeded_differential_matches_full_history(seed):
    run = Lockstep()
    accepted = set()
    for i, op in enumerate(_ops(seed, 120)):
        if op[0] == "restore_old":
            run.restore_old(op[1]["back"])
            continue
        result = run.step(op, cold=(i % 3 == 0))     # alternate warm and cold restore
        if result[0] == "ok":
            accepted.add(op[0])
    assert {"admit", "commit"} <= accepted


# ------------------------------------------------ persistence and integrity
def _run_with_history():
    run = Lockstep()
    accepted = []
    for op in [("admit", {"id": "intent.1", "n": 1, "tick": 1}), _commit("c1", 1, intent="intent.1"),
               ("close", {"id": "c1", "status": "satisfied", "reason": "anticipated_effect_observed",
                          "tick": 2, "outcome": True}),
               _commit("c2", 3), ("close_exploration", {"id": "c2", "tick": 4}), _commit("c3", 5)]:
        accepted.append(run.step(op)[0])
    assert accepted == ["ok"] * 6, accepted
    return run


def test_missing_dependency_is_refused():
    run = _run_with_history()
    payload = run.compact_ckpts[-1]
    store = dict(run.store)
    del store[payload["history"]["head"]]        # checkpoint durable, dependency not
    with pytest.raises(ah.ActionHistoryError, match="unavailable"):
        ah.restore_compact(payload, organism_id=ORG, resolve=store.__getitem__)


def test_corrupted_dependency_is_refused():
    run = _run_with_history()
    payload = run.compact_ckpts[-1]
    store = dict(run.store)
    head = payload["history"]["head"]
    tampered = bytearray(store[head])
    tampered[len(tampered) // 2] ^= 0x01
    store[head] = bytes(tampered)
    with pytest.raises(ah.ActionHistoryError, match="digest"):
        ah.restore_compact(payload, organism_id=ORG, resolve=store.__getitem__)


def test_foreign_or_miscounted_history_is_refused():
    run = _run_with_history()
    payload = copy.deepcopy(run.compact_ckpts[-1])
    with pytest.raises(ah.ActionHistoryError, match="owner"):
        ah.restore_compact(payload, organism_id="organism.other", resolve=run.store.__getitem__)
    payload["history"]["count"] += 1
    with pytest.raises(ah.ActionHistoryError):
        ah.restore_compact(payload, organism_id=ORG, resolve=run.store.__getitem__)


def test_orphan_segment_from_uncommitted_step_is_harmless_and_branches_coexist():
    run = _run_with_history()
    committed = run.compact_ckpts[-1]
    # step A: dependency written, payload never committed (crash)
    domain, index = ah.restore_compact(committed, organism_id=ORG, resolve=run.store.__getitem__)
    apply(domain, ("close_exploration", {"id": "c3", "tick": 6}))
    _, orphan, _ = ah.checkpoint_compact(domain, index, organism_id=ORG)
    run.store[sha256(orphan).hexdigest()] = orphan
    # recovery restores the committed checkpoint and continues on another branch
    domain, index = ah.restore_compact(committed, organism_id=ORG, resolve=run.store.__getitem__)
    apply(domain, ("close_exploration", {"id": "c3", "tick": 7}))
    branch, segment, _ = ah.checkpoint_compact(domain, index, organism_id=ORG)
    run.store[sha256(segment).hexdigest()] = segment
    assert sha256(segment).hexdigest() != sha256(orphan).hexdigest()
    ah.restore_compact(branch, organism_id=ORG, resolve=run.store.__getitem__)
    ah.restore_compact(committed, organism_id=ORG, resolve=run.store.__getitem__)


def test_cold_restore_runs_canonical_validation_on_reconstructed_history():
    run = _run_with_history()
    payload = run.compact_ckpts[-1]
    store = dict(run.store)
    # Rewrite history consistently (valid digests) but with a record that the
    # canonical ActionDomain.restore rejects: a closed commitment whose intent
    # status contradicts it.
    records = ah.load_history(ah.HistoryRef(payload["history"]["head"], payload["history"]["count"]),
                              organism_id=ORG, resolve=store.__getitem__)
    for r in records:
        if r["kind"] == "intent":
            r["record"]["status"] = "failed"
    seg = ah.encode_segment(organism_id=ORG, previous=ah.EMPTY, records=records)
    store[sha256(seg).hexdigest()] = seg
    forged = dict(payload, history={"head": sha256(seg).hexdigest(), "count": len(records)})
    with pytest.raises(ValueError):
        ah.restore_compact(forged, organism_id=ORG, resolve=store.__getitem__)


# ------------------------------------------------------------ v4 conversion
def test_v4_conversion_is_lossless_and_non_destructive():
    run = _run_with_history()
    original = run.oracle.checkpoint()
    frozen = json.dumps(original, sort_keys=True)
    payload, segments = ah.v4_to_compact(original, organism_id=ORG)
    store = {sha256(s).hexdigest(): s for s in segments}
    assert json.dumps(original, sort_keys=True) == frozen            # input untouched
    assert ah.compact_to_v4(payload, organism_id=ORG, resolve=store.__getitem__) == original
    assert len(json.dumps(payload)) < len(frozen)


# ---------------------------------------------------------- v5 declaration
def test_v5_declaration_and_exact_v4_equivalent():
    run = _run_with_history()
    v4 = {"schema": ah.EXPLORATION_V4_SCHEMA, "organism_id": ORG, "generation": 1, "world_id": "w",
          "source_causal_tick": 5, "target_causal_tick": 6, "physical_step_id": "s", "commit_id": "c",
          "transition_state_base64": "", "engine": {}, "action_domain": run.oracle.checkpoint()}
    domain, index = ah.restore_compact(run.compact_ckpts[-1], organism_id=ORG, resolve=run.store.__getitem__)
    parent = index.ref.head
    apply(domain, ("close_exploration", {"id": "c3", "tick": 6}))
    v5, decl, index = ah.exploration_checkpoint_v5(v4, domain, index)
    assert decl.parent == parent and decl.new_blob is not None and decl.head == index.ref.head
    run.store[decl.head] = decl.new_blob
    oracle = ActionDomain.restore(copy.deepcopy(v4["action_domain"]))
    apply(oracle, ("close_exploration", {"id": "c3", "tick": 6}))
    expected_v4 = ah._encode_exploration(dict(v4, action_domain=oracle.checkpoint()))
    assert ah.exploration_v5_to_v4(v5, resolve=run.store.__getitem__) == expected_v4
    # a step that makes nothing terminal declares no new bytes and keeps its head
    _, restored, index = ah.restore_exploration_v5_domain(v5, resolve=run.store.__getitem__)
    _, decl2, _ = ah.exploration_checkpoint_v5(v4, restored, index)
    assert decl2.new_blob is None and decl2.head == decl2.parent == decl.head
    with pytest.raises(ah.ActionHistoryError):
        ah.DependencyDeclaration(decl.head, decl.parent, b"other bytes")


def test_retired_ids_cannot_overlap_live_records():
    live = ActionDomain(outcome_ledger=ExecutiveOutcomeLedger())
    apply(live, _commit("x", 1))
    with pytest.raises(ValueError, match="retired"):
        ActionDomain.restore_live(live.checkpoint(), retired_intent_ids=frozenset(),
                                  retired_commitment_ids=frozenset({"x"}), retired_closed_ids=frozenset())
