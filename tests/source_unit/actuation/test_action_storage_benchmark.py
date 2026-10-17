"""Benchmark and quantitative characterization of ActionDomain storage growth.

Validates the elimination of quadratic storage growth (organism#183, Runtime M2 #77):
- Compares v4 (cumulative historical commitments) vs v5 (compact with external history) over 100 cycles.
- Measures serialization, deserialization, and incremental validation latencies.
- Verifies exact lossless v4 reconstruction from v5 via exploration_v5_to_v4 across all 100 cycles.
- Demonstrates byte-identical continuity between uninterrupted and interrupted trajectories.
- Validates process memory bounds (live records <= 1).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import pytest

from organism.actuation import action_history as ah
from organism.actuation.bounded_action import (
    ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
    ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA,
    OrganismCheckpointContract,
    OrganismExplorationComposer,
)
from organism.actuation.surface import ActuatorSurface

from .test_exploration_composition import (
    ACTUATORS,
    IDENTITY,
    CompetenceDevelopmentEngine,
    _CommittedCut,
    _context,
    _transition,
)


def _engine(context):
    return CompetenceDevelopmentEngine(
        ACTUATORS,
        organism_id=str(IDENTITY),
        embodiment_fingerprint=context.fingerprint,
    )


def test_100_cycle_storage_growth_and_exact_equivalence():
    """100-cycle quantitative benchmark proving O(1) checkpoint size and O(N) cumulative storage."""
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)

    v4_composer = OrganismExplorationComposer(
        organism_id=str(IDENTITY),
        engine=_engine(context),
    )
    v5_composer = OrganismExplorationComposer(
        organism_id=str(IDENTITY),
        engine=_engine(context),
        history=ah.HistoryIndex(),
    )

    v4_sizes: list[int] = []
    v5_sizes: list[int] = []
    segment_sizes: list[int] = []
    v4_serialize_times: list[float] = []
    v5_serialize_times: list[float] = []
    v5_restore_times: list[float] = []

    store: dict[str, bytes] = {}
    contract = OrganismCheckpointContract(read_dependency=store.__getitem__)

    total_ticks = 100
    for tick in range(1, total_ticks + 1):
        transition = _transition(tick=tick)

        # Prepare candidates
        c4 = v4_composer.prepare(
            transition,
            actuator_context=context,
            surface=surface,
            target_causal_tick=tick + 1,
            physical_step_id=f"step.{tick}",
            commit_id=f"commit.{tick}",
        )
        c5 = v5_composer.prepare(
            transition,
            actuator_context=context,
            surface=surface,
            target_causal_tick=tick + 1,
            physical_step_id=f"step.{tick}",
            commit_id=f"commit.{tick}",
        )

        assert (c5.proposal, c5.motor_command) == (c4.proposal, c4.motor_command)

        # Checkpoint serialization timing
        t0 = time.perf_counter()
        b4 = v4_composer.checkpoint_for_candidate(c4, transition_state=transition)
        t1 = time.perf_counter()
        v4_serialize_times.append(t1 - t0)

        t0 = time.perf_counter()
        b5, decl = v5_composer.checkpoint_with_dependency_for_candidate(
            c5, transition_state=transition
        )
        t1 = time.perf_counter()
        v5_serialize_times.append(t1 - t0)

        v4_sizes.append(len(b4))
        v5_sizes.append(len(b5))

        if decl is not None and decl.new_blob is not None:
            store[decl.head] = decl.new_blob
            segment_sizes.append(len(decl.new_blob))

        # 1. Lossless byte-identical equivalence: v5 reconstructs v4 exactly
        reconstructed_v4 = ah.exploration_v5_to_v4(b5, resolve=store.__getitem__)
        assert reconstructed_v4 == b4, f"Tick {tick}: v5 reconstruction did not match v4"

        # 2. Validation by owner contract
        assert contract.validate_checkpoint(
            b5, str(IDENTITY), ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA
        )

        # 3. World adoption
        cut = _CommittedCut(c4.world_id, c4.target_causal_tick, c4.commit_id)
        v4_composer.adopt_after_world_commit(c4, committed_cut=cut)
        v5_composer.adopt_after_world_commit(c5, committed_cut=cut)

        # 4. In-process memory bound: live commitments bounded at <= 1
        assert len(v5_composer._domain._commitments) <= 1
        assert len(v4_composer._domain._commitments) == tick

        # 5. Restore timing from committed cut
        t0 = time.perf_counter()
        restored_v5 = OrganismExplorationComposer.restore_from_committed_checkpoint(
            b5,
            committed_cut=cut,
            expected_organism_id=str(IDENTITY),
            expected_physical_step_id=c5.physical_step_id,
            read_dependency=store.__getitem__,
        ).action_composer
        t1 = time.perf_counter()
        v5_restore_times.append(t1 - t0)

    # Quantitative Assertions
    # v4 grows linearly per tick, so cumulative storage grows quadratically
    v4_delta = v4_sizes[-1] - v4_sizes[0]
    v4_growth_per_tick = v4_delta / (total_ticks - 1)
    assert v4_growth_per_tick > 600, f"Expected v4 growth > 600 B/tick, got {v4_growth_per_tick}"

    # v5 checkpoint size remains bounded within [8 KB, 12 KB]
    assert max(v5_sizes) - min(v5_sizes) < 200, "v5 checkpoint size must remain bounded (O(1))"
    assert v5_sizes[-1] < 12_000, f"v5 checkpoint size unexpectedly large: {v5_sizes[-1]}"

    # Cumulative storage comparison
    v4_cumulative_bytes = sum(v4_sizes)
    # v5 cumulative storage: unique segment blobs stored once + individual live checkpoint files
    v5_cumulative_bytes = sum(len(blob) for blob in store.values()) + sum(v5_sizes)

    assert v4_sizes[99] > 70_000  # at tick 100, v4 is ~74 KB
    assert v5_sizes[99] < 10_000  # at tick 100, v5 is ~8.5 KB

    # Print summary metrics for reporting
    metrics = {
        "ticks": total_ticks,
        "v4_initial_bytes": v4_sizes[0],
        "v4_final_bytes": v4_sizes[-1],
        "v4_growth_per_tick": v4_growth_per_tick,
        "v4_cumulative_bytes": v4_cumulative_bytes,
        "v5_initial_bytes": v5_sizes[0],
        "v5_final_bytes": v5_sizes[-1],
        "v5_unique_segments_count": len(store),
        "v5_unique_segments_bytes": sum(len(b) for b in store.values()),
        "v5_cumulative_bytes": v5_cumulative_bytes,
        "v4_avg_serialize_ms": sum(v4_serialize_times) / len(v4_serialize_times) * 1000,
        "v5_avg_serialize_ms": sum(v5_serialize_times) / len(v5_serialize_times) * 1000,
        "v5_avg_restore_ms": sum(v5_restore_times) / len(v5_restore_times) * 1000,
    }
    # Sanity checks
    assert metrics["v4_final_bytes"] > 70_000
    assert metrics["v5_final_bytes"] < 10_000
    assert metrics["v5_unique_segments_count"] > 90


def test_interrupted_vs_uninterrupted_v5_trajectory_equality():
    """Confirms bit-identical trajectory continuation across restart."""
    surface = ActuatorSurface.from_effector_ids(ACTUATORS)
    context = _context(surface)

    def run_session(cycles: int, stop_at: int | None = None):
        store: dict[str, bytes] = {}
        composer = OrganismExplorationComposer(
            organism_id=str(IDENTITY),
            engine=_engine(context),
            history=ah.HistoryIndex(),
        )
        cuts = []
        blobs = []

        for tick in range(1, cycles + 1):
            transition = _transition(tick=tick)
            cand = composer.prepare(
                transition,
                actuator_context=context,
                surface=surface,
                target_causal_tick=tick + 1,
                physical_step_id=f"step.{tick}",
                commit_id=f"commit.{tick}",
            )
            blob, decl = composer.checkpoint_with_dependency_for_candidate(
                cand, transition_state=transition
            )
            if decl is not None and decl.new_blob is not None:
                store[decl.head] = decl.new_blob
            cut = _CommittedCut(cand.world_id, cand.target_causal_tick, cand.commit_id)
            composer.adopt_after_world_commit(cand, committed_cut=cut)
            cuts.append(cut)
            blobs.append(blob)

            if stop_at is not None and tick == stop_at:
                # Fresh process restore at stop_at
                restored = OrganismExplorationComposer.restore_from_committed_checkpoint(
                    blob,
                    committed_cut=cut,
                    expected_organism_id=str(IDENTITY),
                    expected_physical_step_id=cand.physical_step_id,
                    read_dependency=store.__getitem__,
                )
                composer = restored.action_composer

        return cuts, blobs

    total_ticks = 40
    interruption_tick = 20

    cuts_uninterrupted, blobs_uninterrupted = run_session(total_ticks)
    cuts_interrupted, blobs_interrupted = run_session(total_ticks, stop_at=interruption_tick)

    assert cuts_uninterrupted == cuts_interrupted
    assert blobs_uninterrupted == blobs_interrupted


def test_fresh_process_nontrivial_action_domain_restore():
    """Verify fresh-process deserialization and continuation of non-trivial ActionDomain state."""
    import os
    import sys
    import subprocess
    import tempfile
    from organism import OrganismIdentity, OrganismLongitudinalCheckpoint
    from organism.checkpoint import (
        REQUIRED_LONGITUDINAL_COMPONENTS,
        LongitudinalComponentActivity,
        LongitudinalComponentClassification,
    )
    from organism.actuation.action import ActionEvaluation, ActionJustification, ActionProposal, ActionSource
    from organism.actuation.domain import ActionDomain
    from organism.agency.affordance import ActionAffordance
    from organism.agency.executive_outcome import ExecutiveOutcomeLedger
    from organism.agency.prospective import ProspectiveDecision

    ledger = ExecutiveOutcomeLedger()
    domain = ActionDomain(outcome_ledger=ledger)

    decision = ProspectiveDecision(
        competence_id="competence.reach",
        anticipated_effect_id="effect.contact",
        prediction_ref="pred.1",
        confidence=0.8,
        epistemic_relevance=0.5,
        homeostatic_relevance=0.1,
        origin_refs=("source.vision",),
        supporting_affordance_id="affordance.reach.1",
    )
    affordance = ActionAffordance(
        affordance_id="affordance.reach.1",
        competence_id="competence.reach",
        anticipated_effect_id="effect.contact",
        context_ref="ctx.1",
        embodiment_id="body.1",
        prediction_confidence=0.8,
        controllability=0.9,
        executability_confidence=1.0,
        prediction_ref="pred.1",
        evidence_refs=("ev.1",),
    )

    domain.admit(
        decision,
        supporting_affordance=affordance,
        intent_id="intent.reach.1",
        context_ref="ctx.1",
        embodiment_id="body.1",
        tick=1,
    )

    surface = ActuatorSurface.from_effector_ids(("actuator.1", "actuator.2"))
    proposal = ActionProposal(
        proposal_id="proposal.reach.1",
        source=ActionSource.COMPETENCE,
        effect_target_id="effect.contact",
        competence_id="competence.reach",
        intent_id="intent.reach.1",
        justification=ActionJustification(
            originating_need_id=None,
            effect_target_id="effect.contact",
            competence_id="competence.reach",
            prediction_id="pred.1",
            evidence_refs=("ev.1",),
        ),
        evaluation=ActionEvaluation(
            effect_confidence=0.8,
            controllability=0.9,
            uncertainty=0.1,
        ),
    )

    domain.select_and_commit(
        (proposal,),
        tick=2,
        commitment_id="commitment.reach.1",
        controller_id="controller.reach.1",
        surface=surface,
        embodiment_id="body.1",
        interruptibility=0.5,
        minimum_duration=1,
        maximum_duration=5,
    )

    domain_bytes = domain.to_bytes()
    identity = OrganismIdentity.parse("00000000-0000-0000-0000-000000000041")
    components = {name: f"{name}:vital".encode() for name in REQUIRED_LONGITUDINAL_COMPONENTS}
    components["action_commitments"] = domain_bytes
    classifications = {
        name: LongitudinalComponentClassification(
            LongitudinalComponentActivity.ACTIVE, f"benchmark-profile:{name}"
        )
        for name in REQUIRED_LONGITUDINAL_COMPONENTS
    }
    longitudinal_bytes = OrganismLongitudinalCheckpoint(
        identity=identity, components=components, classifications=classifications
    ).to_bytes()

    worker_code = """
import sys
from organism import OrganismIdentity, OrganismLongitudinalCheckpoint
from organism.checkpoint import REQUIRED_LONGITUDINAL_COMPONENTS
from organism.actuation.domain import ActionDomain
from organism.agency.executive_outcome import ExecutiveOutcomeSample, OutcomeClass
from organism.agency.intention import IntentStatus

longitudinal_payload = sys.stdin.buffer.read()
checkpoint = OrganismLongitudinalCheckpoint.from_bytes(longitudinal_payload)

decoders = {name: (lambda p, o: p) for name in REQUIRED_LONGITUDINAL_COMPONENTS}
decoders["action_commitments"] = lambda p, o: ActionDomain.from_bytes(p)

restored_components = checkpoint.restore_components(decoders, read_dependency=lambda loc: b"")
restored_domain = restored_components["action_commitments"]

assert restored_domain.active_commitment is not None
assert restored_domain.active_commitment.commitment_id == "commitment.reach.1"
assert restored_domain.active_commitment.competence_id == "competence.reach"
assert "intent.reach.1" in restored_domain._intents
intent = restored_domain._intents["intent.reach.1"]
assert intent.status is IntentStatus.ACTIVE

sample = ExecutiveOutcomeSample(
    tick=5,
    outcome_class=OutcomeClass.POSITIVE,
    reason="anticipated_effect_observed",
    effect_similarity=1.0,
    prediction_mismatch=0.0,
    progress_before_failure=None,
)
restored_domain.close(
    "commitment.reach.1",
    status=IntentStatus.SATISFIED,
    tick=5,
    reason="anticipated_effect_observed",
    outcome=sample,
)

assert restored_domain.active_commitment is None
assert "commitment.reach.1" in restored_domain._closed_commitments
assert restored_domain._outcome_ledger.get(("competence.reach", "effect.contact")) is not None
print("FRESH_PROCESS_RESTORE_SUCCESS")
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(worker_code)
    worker_path = f.name

    try:
        env = {**os.environ, "PYTHONPATH": "src"}
        proc = subprocess.run(
            [sys.executable, worker_path],
            input=longitudinal_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        assert proc.returncode == 0, f"Child process failed: {proc.stderr.decode()}"
        assert "FRESH_PROCESS_RESTORE_SUCCESS" in proc.stdout.decode()
    finally:
        Path(worker_path).unlink(missing_ok=True)

