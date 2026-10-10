from __future__ import annotations

import hashlib
import random

import pytest

import organism.actuation.sensorimotor as sensorimotor_module
from organism.actuation.competence import CompetenceMaturity
from organism.actuation.sensorimotor import CompetenceDevelopmentEngine, MotorPrimitive


def _ids(count: int = 8) -> tuple[str, ...]:
    return tuple(f"actuator.{index}" for index in range(count))


def _test_embodiment_fingerprint(organism_id: str, actuator_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256("|".join(actuator_ids).encode("utf-8")).hexdigest()
    return f"test-embodiment:{organism_id}:{digest}"


def test_exploration_covers_all_actuators_without_single_channel_monopoly():
    learner = CompetenceDevelopmentEngine(
        _ids(),
        organism_id="org-babble",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-babble", _ids()),
        max_concurrent=4,
    )

    seen = set()
    for tick in range(128):
        intents = learner.motor_intents(tick)
        assert len(intents) <= 4
        seen.update(intent.actuator_id for intent in intents)

    assert seen == set(_ids())
    assert learner.exploration_coverage == 1.0


def test_exploration_holds_channel_set_within_short_epoch():
    learner = CompetenceDevelopmentEngine(
        _ids(),
        organism_id="org-hold",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-hold", _ids()),
        max_concurrent=4,
    )

    sets = []
    for tick in range(1, 7):
        intents = learner.motor_intents(tick)
        sets.append({intent.actuator_id for intent in intents})

    assert sets
    assert all(current == sets[0] for current in sets[1:])


def test_spontaneous_motor_bout_spans_multiple_independent_evidence_blocks():
    learner = CompetenceDevelopmentEngine(
        _ids(),
        organism_id="org-bout-persistence",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-bout-persistence", _ids()),
        max_concurrent=4,
    )

    samples = {}
    for tick in range(24):
        intents = learner.motor_intents(tick)
        if tick in {7, 15, 23}:
            samples[tick] = {intent.actuator_id for intent in intents}

    assert samples[7]
    assert samples[7] == samples[15] == samples[23]


def test_autonomous_babbling_can_recur_across_independent_evidence_blocks():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-natural-recurrence",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-natural-recurrence", _ids(4)),
        max_concurrent=2,
    )

    state = {"sense.a": 0.0, "sense.b": 0.0}
    previous_vector = {}
    for tick in range(96):
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector=previous_vector,
            discovery_eligible=True,
        )
        intents = learner.motor_intents(tick)
        previous_vector = {intent.actuator_id: intent.activation for intent in intents}
        drive = sum(previous_vector.values())
        state = {
            "sense.a": state["sense.a"] + drive * 0.01,
            "sense.b": state["sense.b"] - drive * 0.004,
        }

    snapshot = learner.snapshot()
    assert snapshot.recurrent_competence_candidates > 0
    assert snapshot.max_competence_samples >= 2


def test_multi_horizon_statistics_are_recorded_independently():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-horizons",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-horizons", _ids(4)),
        max_concurrent=4,
    )

    state = {f"sense.{i}": 0.0 for i in range(4)}
    previous_vector = {}
    for tick in range(90):
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector=previous_vector,
        )
        intents = learner.motor_intents(tick)
        previous_vector = {intent.actuator_id: intent.activation for intent in intents}
        # Deterministic body response to the current pattern.
        drive = sum(previous_vector.values()) / max(1, len(previous_vector))
        state = {
            key: value + drive * (0.001 + index * 0.0001)
            for index, (key, value) in enumerate(state.items())
        }

    snapshot = learner.snapshot()
    counts = dict(snapshot.horizon_samples)
    assert counts[1] > 0
    assert counts[4] > 0
    assert counts[16] > 0
    assert counts[64] > 0
    assert snapshot.known_patterns > 0


def _teach_repeated_sequence(
    learner: CompetenceDevelopmentEngine,
    *,
    episodes: int = 2,
) -> None:
    sequence = (
        {"actuator.0": 0.7, "actuator.1": 0.3},
        {"actuator.0": 0.5, "actuator.2": 0.6},
        {"actuator.1": 0.6, "actuator.3": 0.4},
        {"actuator.0": 0.3, "actuator.2": 0.7, "actuator.3": 0.2},
    )
    state = {"sense.a": 0.0, "sense.b": 0.0}
    tick = 0
    for episode_index in range(episodes):
        for step, vector in enumerate(sequence):
            learner.observe(
                tick=tick,
                body_state=state,
                motor_vector=vector,
                discovery_eligible=True,
            )
            drive = sum(vector.values())
            signed = (step + 1) / len(sequence)
            state = {
                "sense.a": state["sense.a"] + drive * 0.01 * signed,
                "sense.b": state["sense.b"] - drive * 0.006 * signed,
            }
            tick += 1
        # This next pre-action state closes the causal four-action episode.
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector={},
            discovery_eligible=False,
        )
        tick += 1
        # Independent recurrence must come from a later exploration block. Pad
        # only between episodes; padding after the final episode would clear
        # last_natural_competence_ids, which is intentionally one-tick evidence.
        if episode_index + 1 < episodes:
            while tick % 8:
                learner.observe(
                    tick=tick,
                    body_state=state,
                    motor_vector={},
                    discovery_eligible=False,
                )
                tick += 1


def test_single_episode_remains_candidate_until_independent_recurrence():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-single-episode-candidate",
        embodiment_fingerprint=_test_embodiment_fingerprint(
            "org-single-episode-candidate", _ids(4)
        ),
        max_concurrent=4,
    )
    sequence = (
        (("actuator.0", 5), ("actuator.1", 3)),
        (("actuator.0", 4), ("actuator.2", 4)),
        (("actuator.1", 5), ("actuator.3", 2)),
        (("actuator.0", 3), ("actuator.2", 5)),
    )

    result = learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=4,
        may_create=True,
    )

    assert result is None
    assert learner.primitives == ()
    snapshot = learner.snapshot()
    assert snapshot.competence_candidates == 1
    assert snapshot.recurrent_competence_candidates == 0
    lifecycle = learner.checkpoint()["primitive_stats"][0]
    assert lifecycle["first_sample_tick"] == 4
    assert lifecycle["last_sample_tick"] == 4
    assert lifecycle["materialized_tick"] is None
    assert lifecycle["competence_tick"] is None


def test_checkpoint_controller_presence_does_not_create_established_competence():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-controller-only-checkpoint",
        embodiment_fingerprint=_test_embodiment_fingerprint(
            "org-controller-only-checkpoint", _ids(4)
        ),
        max_concurrent=4,
    )
    sequence = (
        (("actuator.0", 5),),
        (("actuator.1", 5),),
        (("actuator.2", 5),),
        (("actuator.3", 5),),
    )
    learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=4,
        may_create=True,
        evidence_blocks=frozenset({0}),
    )

    payload = learner.checkpoint()
    assert payload["schema_version"] == 12
    assert payload["primitives"] == []

    restored = CompetenceDevelopmentEngine.restore(
        payload,
        actuator_ids=_ids(4),
        organism_id="org-controller-only-checkpoint",
        embodiment_fingerprint=_test_embodiment_fingerprint(
            "org-controller-only-checkpoint", _ids(4)
        ),
    )
    assert restored.snapshot().competence_chunks == 0
    assert restored.snapshot().established_competences == 0


def test_adjacent_windows_from_same_exploration_block_do_not_count_as_recurrence():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-same-evidence-block",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-same-evidence-block", _ids(4)),
        max_concurrent=4,
    )
    sequence = (
        (("actuator.0", 5),),
        (("actuator.1", 5),),
        (("actuator.2", 5),),
        (("actuator.3", 5),),
    )

    learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=4,
        may_create=True,
        evidence_blocks=frozenset({0}),
    )
    learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=8,
        may_create=True,
        evidence_blocks=frozenset({0}),
    )

    assert learner.snapshot().recurrent_competence_candidates == 0
    assert learner.primitives == ()


def test_same_sequence_in_disjoint_exploration_block_counts_as_recurrence():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-independent-evidence-block",
        embodiment_fingerprint=_test_embodiment_fingerprint(
            "org-independent-evidence-block", _ids(4)
        ),
        max_concurrent=4,
    )
    sequence = (
        (("actuator.0", 5),),
        (("actuator.1", 5),),
        (("actuator.2", 5),),
        (("actuator.3", 5),),
    )

    learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=4,
        may_create=True,
        evidence_blocks=frozenset({0}),
    )
    learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=12,
        may_create=True,
        evidence_blocks=frozenset({1}),
    )

    assert learner.primitives
    assert learner.primitives[0].samples == 2


def test_reproducible_temporal_sequence_can_mature_motor_competence():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-primitive",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-primitive", _ids(4)),
        max_concurrent=4,
    )

    _teach_repeated_sequence(learner, episodes=2)

    snapshot = learner.snapshot()
    assert snapshot.competence_chunks > 0
    assert snapshot.best_controllability > 0.0
    assert learner.primitives
    assert learner.cognitive_primitives
    lifecycle_items = learner.checkpoint()["primitive_stats"]
    promoted = [item for item in lifecycle_items if item["materialized_tick"] is not None]
    assert promoted
    assert all(item["first_sample_tick"] <= item["materialized_tick"] for item in promoted)
    competent = [item for item in promoted if item["competence_tick"] is not None]
    assert competent
    assert all(item["materialized_tick"] <= item["competence_tick"] for item in competent)


def test_sensorimotor_checkpoint_roundtrip_preserves_learning_state():
    learner = CompetenceDevelopmentEngine(
        _ids(6),
        organism_id="org-restore",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-restore", _ids(6)),
        max_concurrent=4,
    )

    state = {"sense.a": 0.0}
    previous_vector = {}
    for tick in range(100):
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector=previous_vector,
        )
        intents = learner.motor_intents(tick)
        previous_vector = {intent.actuator_id: intent.activation for intent in intents}
        state["sense.a"] += sum(previous_vector.values()) * 0.001

    restored = CompetenceDevelopmentEngine.restore(
        learner.checkpoint(),
        actuator_ids=_ids(6),
        organism_id="org-restore",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-restore", _ids(6)),
    )

    assert restored.snapshot() == learner.snapshot()
    assert restored.primitives == learner.primitives
    assert tuple(item.maturity for item in restored.cognitive_primitives) == tuple(
        item.maturity for item in learner.cognitive_primitives
    )
    assert all(item.competence_evidence.support >= 2 for item in restored.cognitive_primitives)


def test_checkpoint_restore_preserves_identity_and_effective_concurrency_output():
    ids = _ids(6)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-checkpoint-config",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-checkpoint-config", ids),
        max_concurrent=1,
    )
    for tick in range(17):
        learner.motor_intents(tick)

    payload = learner.checkpoint()
    assert payload["schema_version"] == 12
    assert payload["organism_id"] == "org-checkpoint-config"
    assert payload["max_concurrent"] == 1

    restored = CompetenceDevelopmentEngine.restore(
        payload,
        actuator_ids=ids,
        organism_id="org-checkpoint-config",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-checkpoint-config", ids),
        expected_max_concurrent=1,
    )
    for tick in range(17, 17 + 24 * 8):
        assert restored.motor_intents(tick) == learner.motor_intents(tick)


def test_restore_rejects_checkpoint_for_another_organism():
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-original",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-original", ids),
    )

    with pytest.raises(ValueError, match="organism identity mismatch"):
        CompetenceDevelopmentEngine.restore(
            learner.checkpoint(),
            actuator_ids=ids,
            organism_id="org-other",
            embodiment_fingerprint=_test_embodiment_fingerprint("org-original", ids),
        )


def test_restore_rejects_max_concurrent_mismatch_and_invalid_checkpoint_value():
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-config-mismatch",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-config-mismatch", ids),
        max_concurrent=1,
    )
    payload = learner.checkpoint()

    with pytest.raises(ValueError, match="max_concurrent mismatch"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id="org-config-mismatch",
            embodiment_fingerprint=_test_embodiment_fingerprint("org-config-mismatch", ids),
            expected_max_concurrent=2,
        )

    payload["max_concurrent"] = 0
    with pytest.raises(ValueError, match="max_concurrent"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id="org-config-mismatch",
            embodiment_fingerprint=_test_embodiment_fingerprint("org-config-mismatch", ids),
        )


def test_cognitive_primitive_replays_only_learned_motor_pattern():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-cognitive-primitive",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-cognitive-primitive", _ids(4)),
        max_concurrent=4,
    )
    _teach_repeated_sequence(learner, episodes=2)

    primitives = learner.cognitive_primitives
    assert primitives
    primitive = primitives[0]
    intents = learner.primitive_intents(primitive.primitive_id)

    assert intents
    assert {intent.actuator_id for intent in intents} == {
        actuator_id for actuator_id, level in primitive.sequence[0] if level > 0
    }
    assert len(primitive.sequence) == 4
    assert len(set(primitive.sequence)) > 1
    assert learner.primitive_intents("primitive.not-learned") == ()


def test_cognitive_primitive_execution_preserves_full_temporal_duration():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-atomic-primitive",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-atomic-primitive", _ids(4)),
        max_concurrent=4,
    )
    _teach_repeated_sequence(learner, episodes=2)
    primitive = learner.cognitive_primitives[0]

    assert learner.activate_primitive(primitive.primitive_id)

    outputs = []
    for tick in range(primitive.duration_ticks):
        intents = learner.motor_intents(10_000 + tick)
        outputs.append(
            tuple((intent.actuator_id, round(intent.activation, 6)) for intent in intents)
        )
        assert learner.last_output_source == "primitive"
        assert learner.last_output_primitive_id == primitive.primitive_id

    expected = [
        tuple((actuator_id, round(level / 7.0, 6)) for actuator_id, level in pattern if level > 0)
        for pattern in primitive.sequence
    ]
    assert outputs == expected
    assert len(set(outputs)) > 1
    assert learner.active_primitive_id is None


def test_inconsistent_repetition_retracts_false_motor_primitive():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-falsify-primitive",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-falsify-primitive", _ids(4)),
        max_concurrent=4,
    )
    sequence = (
        {"actuator.0": 0.7, "actuator.1": 0.3},
        {"actuator.0": 0.5, "actuator.2": 0.6},
        {"actuator.1": 0.6, "actuator.3": 0.4},
        {"actuator.0": 0.3, "actuator.2": 0.7},
    )
    state = {"sense.a": 0.0}
    tick = 0

    for sign in (1.0, -1.0):
        for vector in sequence:
            learner.observe(
                tick=tick,
                body_state=state,
                motor_vector=vector,
                discovery_eligible=True,
            )
            state["sense.a"] += sign * sum(vector.values()) * 0.01
            tick += 1
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector={},
            discovery_eligible=False,
        )
        tick += 1

    assert learner.cognitive_primitives == ()


def test_exploration_can_discover_temporal_chunk_across_synergy_boundary():
    learner = CompetenceDevelopmentEngine(
        _ids(8),
        organism_id="org-cross-synergy",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-cross-synergy", _ids(8)),
        max_concurrent=4,
    )

    state = {"sense.a": 0.0, "sense.b": 0.0}
    for tick in range(96):
        intents = learner.motor_intents(tick)
        vector = {intent.actuator_id: intent.activation for intent in intents}
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector=vector,
            discovery_eligible=True,
        )
        weighted = sum(
            (index + 1) * value
            for index, (_actuator_id, value) in enumerate(sorted(vector.items()))
        )
        state = {
            "sense.a": state["sense.a"] + weighted * 0.002,
            "sense.b": state["sense.b"] - weighted * 0.001,
        }

    # Close the final causal window.
    learner.observe(
        tick=96,
        body_state=state,
        motor_vector={},
        discovery_eligible=False,
    )

    checkpoint = learner.checkpoint()
    assert checkpoint["primitive_stats"]
    assert any(
        len({tuple((item[0], item[1]) for item in pattern) for pattern in candidate["sequence"]})
        > 1
        for candidate in checkpoint["primitive_stats"]
    )


def test_passive_drift_is_subtracted_from_motor_controllability():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-passive-baseline",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-passive-baseline", _ids(4)),
        max_concurrent=4,
    )

    state = {"sense.a": 0.0}
    # Learn one four-tick passive drift baseline of +0.04.
    for tick in range(4):
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector={},
            discovery_eligible=False,
        )
        state["sense.a"] += 0.01
    learner.observe(
        tick=4,
        body_state=state,
        motor_vector={},
        discovery_eligible=False,
    )
    assert learner.snapshot().passive_baseline_samples >= 1

    # Apply a motor sequence while the body changes by exactly the same amount.
    sequence = (
        {"actuator.0": 0.6},
        {"actuator.1": 0.6},
        {"actuator.2": 0.6},
        {"actuator.3": 0.6},
    )
    tick = 6
    for vector in sequence:
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector=vector,
            discovery_eligible=True,
        )
        state["sense.a"] += 0.01
        tick += 1
    learner.observe(
        tick=tick,
        body_state=state,
        motor_vector={},
        discovery_eligible=False,
    )

    assert learner.primitives == ()


@pytest.mark.parametrize(
    "mutator",
    (
        lambda payload: payload.update({"schema_version": True}),
        lambda payload: payload.update({"smoothing": float("nan")}),
        lambda payload: payload["primitives"][0].update({"samples": "2"}),
        lambda payload: payload["primitives"][0].update(
            {"samples": payload["primitives"][0]["samples"] + 1}
        ),
    ),
)
def test_sensorimotor_restore_rejects_coerced_or_nonfinite_skill_state(mutator):
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-strict-restore",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-strict-restore", _ids(4)),
        max_concurrent=4,
    )
    _teach_repeated_sequence(learner, episodes=2)
    payload = learner.checkpoint()
    assert payload["primitives"]

    mutator(payload)

    with pytest.raises(ValueError):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=_ids(4),
            organism_id="org-strict-restore",
        )


@pytest.mark.parametrize(
    "lifecycle_mutator",
    (
        lambda item: item.pop("first_sample_tick"),
        lambda item: item.update({"last_sample_tick": item["first_sample_tick"] - 1}),
        lambda item: item.update({"competence_tick": 1, "materialized_tick": None}),
    ),
)
def test_sensorimotor_restore_rejects_corrupted_primitive_lifecycle(lifecycle_mutator):
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-lifecycle-restore",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-lifecycle-restore", _ids(4)),
        max_concurrent=4,
    )
    _teach_repeated_sequence(learner, episodes=2)
    payload = learner.checkpoint()
    lifecycle_mutator(payload["primitive_stats"][0])

    with pytest.raises(ValueError):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=_ids(4),
            organism_id="org-lifecycle-restore",
        )


@pytest.mark.parametrize("legacy_schema", [1, 2, 3, 4, 5, 6, 7, 8])
def test_restore_rejects_every_pre_v9_schema_outright(legacy_schema):
    """Pre-v9 checkpoints cannot be represented honestly by the current learner.

    They may carry removed verification apparatus, body-wide cardinality bias,
    density-biased recurrence evidence, or no mutually-exclusive motor-unit
    constitution.  Reinterpreting that learned history would be scientifically
    dishonest, so restoration fails closed.
    """
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-legacy-schema",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-legacy-schema", _ids(4)),
    )
    payload = learner.checkpoint()
    payload["schema_version"] = legacy_schema

    with pytest.raises(ValueError):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=_ids(4),
            organism_id="org-legacy-schema",
        )


def test_restore_rejects_in_flight_verification_replay_instead_of_relabeling_it():
    """An action the removed scheduler forced must never resurface as
    cognition-originated after restore — that would rewrite the organism's
    own causal history (not merely legacy debt)."""
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-verification-reject",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-verification-reject", _ids(4)),
    )
    _teach_repeated_sequence(learner, episodes=2)
    primitive = learner.cognitive_primitives[0]
    assert learner.activate_primitive(primitive.primitive_id)

    payload = learner.checkpoint()
    assert payload["replay_id"] == primitive.primitive_id
    payload["replay_source"] = "verification"

    with pytest.raises(ValueError):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=_ids(4),
            organism_id="org-verification-reject",
        )


def test_restore_rejects_in_flight_replay_with_missing_or_unknown_source():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-missing-source",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-missing-source", _ids(4)),
    )
    _teach_repeated_sequence(learner, episodes=2)
    primitive = learner.cognitive_primitives[0]
    assert learner.activate_primitive(primitive.primitive_id)

    payload = learner.checkpoint()
    del payload["replay_source"]

    with pytest.raises(ValueError):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=_ids(4),
            organism_id="org-missing-source",
        )


def test_default_exploration_prefers_low_dimensional_coordination_without_forbidding_broad_patterns():
    learner = CompetenceDevelopmentEngine(
        _ids(62),
        organism_id="org-variable-cardinality",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-variable-cardinality", _ids(62)),
    )

    sizes = [len(learner.motor_intents(epoch * 8)) for epoch in range(512)]

    assert min(sizes) >= 1
    assert max(sizes) <= 62
    assert len(set(sizes)) > 8
    ordered = sorted(sizes)
    median = ordered[len(ordered) // 2]
    assert median <= 10
    assert sum(size <= 10 for size in sizes) > len(sizes) / 2
    assert any(size > 31 for size in sizes)


def test_cognitive_primitives_are_not_arbitrarily_truncated_to_eight():
    learner = CompetenceDevelopmentEngine(
        _ids(12),
        organism_id="org-many-competences",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-many-competences", _ids(12)),
    )

    from organism.actuation.sensorimotor import MotorPrimitive

    for index in range(12):
        actuator_id = f"actuator.{index}"
        sequence = tuple(((actuator_id, 5),) for _ in range(4))
        primitive = MotorPrimitive(
            primitive_id=f"primitive.test.{index}",
            embodiment_fingerprint=learner.embodiment_fingerprint,
            sequence=sequence,
            samples=3,
            effect_mean=0.1,
            effect_variance=0.0,
            controllability=0.1 + index * 0.001,
            directional_consistency=1.0,
        )
        learner._primitives[primitive.primitive_id] = primitive

    assert len(learner.cognitive_primitives) == 12


def test_noncontiguous_temporal_window_is_ignored_without_error():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-noncontiguous-window",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-noncontiguous-window", _ids(4)),
        max_concurrent=4,
    )
    state = {"sense.a": 0.0}

    for tick in range(4):
        learner.observe(
            tick=tick,
            body_state=state,
            motor_vector={},
            discovery_eligible=False,
        )
        state["sense.a"] += 0.01

    # Skip tick 4 deliberately. The incomplete temporal window must be ignored,
    # not interpreted as passive/motor evidence and not raise.
    learner.observe(
        tick=5,
        body_state=state,
        motor_vector={},
        discovery_eligible=False,
    )

    assert learner.snapshot().passive_baseline_samples == 0


def test_primitive_ordered_views_are_cached_and_invalidated_on_update():
    learner = CompetenceDevelopmentEngine(
        ("a", "b"),
        organism_id="cache-test",
        embodiment_fingerprint=_test_embodiment_fingerprint("cache-test", ("a", "b")),
    )
    first = MotorPrimitive(
        primitive_id="primitive.b",
        embodiment_fingerprint=learner.embodiment_fingerprint,
        sequence=((("a", 1),),) * 4,
        samples=2,
        effect_mean=0.1,
        effect_variance=0.0,
        controllability=0.1,
        directional_consistency=1.0,
    )
    second = MotorPrimitive(
        primitive_id="primitive.a",
        embodiment_fingerprint=learner.embodiment_fingerprint,
        sequence=((("b", 1),),) * 4,
        samples=2,
        effect_mean=0.1,
        effect_variance=0.0,
        controllability=0.2,
        directional_consistency=1.0,
    )
    learner._primitives[first.primitive_id] = first
    learner._primitives[second.primitive_id] = second
    learner._invalidate_primitive_caches()

    ordered_a = learner.primitives
    ordered_b = learner.primitives
    cognitive_a = learner.cognitive_primitives
    cognitive_b = learner.cognitive_primitives

    assert ordered_a is ordered_b
    assert cognitive_a is cognitive_b
    assert [item.primitive_id for item in ordered_a] == ["primitive.a", "primitive.b"]

    replacement = MotorPrimitive(
        primitive_id="primitive.a",
        embodiment_fingerprint=learner.embodiment_fingerprint,
        sequence=second.sequence,
        samples=3,
        effect_mean=0.2,
        effect_variance=0.0,
        controllability=0.3,
        directional_consistency=1.0,
    )
    learner._primitives[replacement.primitive_id] = replacement
    learner._invalidate_primitive_caches()

    assert learner.primitives is not ordered_a
    assert learner.cognitive_primitives is not cognitive_a
    assert learner.primitives[0] is replacement


def test_observe_precomputes_the_exact_canonical_motor_pattern():
    learner = CompetenceDevelopmentEngine(
        ("b", "a"),
        organism_id="frame-pattern",
        embodiment_fingerprint=_test_embodiment_fingerprint("frame-pattern", ("b", "a")),
    )
    vector = {"b": 0.51, "a": 0.09}
    learner.observe(tick=1, body_state={"x": 0.0}, motor_vector=vector)

    frame = learner._frames[-1]
    assert frame.motor_pattern == (
        ("a", round(0.09 * 7)),
        ("b", round(0.51 * 7)),
    )


def test_primitive_id_cache_preserves_sha256_identity_and_reuses_result():
    learner = CompetenceDevelopmentEngine(
        ("a",),
        organism_id="primitive-id-cache",
        embodiment_fingerprint=_test_embodiment_fingerprint("primitive-id-cache", ("a",)),
    )
    sequence = (
        (("a", 1),),
        (("a", 2),),
        (("a", 3),),
        (("a", 4),),
    )
    expected = "primitive." + hashlib.sha256(repr(sequence).encode("utf-8")).hexdigest()[:16]

    first = learner._primitive_id_for_sequence(sequence)
    second = learner._primitive_id_for_sequence(sequence)

    assert first == expected
    assert second == expected
    assert learner._primitive_id_by_sequence == {sequence: expected}


def test_natural_recurrence_surfaces_competence_without_forced_replay():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-natural-competence",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-natural-competence", _ids(4)),
        max_concurrent=4,
    )

    _teach_repeated_sequence(learner, episodes=2)

    assert learner.cognitive_primitives
    assert learner.last_natural_competence_ids
    assert set(learner.last_natural_competence_ids).issubset(
        {primitive.primitive_id for primitive in learner.cognitive_primitives}
    )
    assert learner.active_primitive_id is None


def test_similar_natural_chunks_count_as_recurrence_not_new_skill():
    learner = CompetenceDevelopmentEngine(
        ("a", "b"),
        organism_id="approx-recurrence",
        embodiment_fingerprint=_test_embodiment_fingerprint("approx-recurrence", ("a", "b")),
    )
    first = (
        (("a", 4), ("b", 2)),
        (("a", 5), ("b", 2)),
        (("a", 5), ("b", 3)),
        (("a", 4), ("b", 3)),
    )
    second = (
        (("a", 4), ("b", 2)),
        (("a", 4), ("b", 2)),
        (("a", 5), ("b", 3)),
        (("a", 4), ("b", 3)),
    )

    learner._record_primitive_episode(
        sequence=first,
        before={"sense.x": 0.0},
        after={"sense.x": 0.05},
        end_tick=4,
        may_create=True,
        evidence_blocks=frozenset({0}),
    )
    learner._record_primitive_episode(
        sequence=second,
        before={"sense.x": 0.0},
        after={"sense.x": 0.05},
        end_tick=12,
        may_create=True,
        evidence_blocks=frozenset({1}),
    )

    assert len(learner.primitives) == 1
    primitive = learner.primitives[0]
    assert primitive.samples == 2
    assert primitive.maturity in {
        CompetenceMaturity.ESTABLISHED,
        CompetenceMaturity.ROBUST,
    }


def test_bounded_primitive_pool_preserves_proven_competence():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-retain-competence",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-retain-competence", _ids(4)),
        max_concurrent=4,
    )

    sequence = (
        (("actuator.0", 4),),
        (("actuator.1", 4),),
        (("actuator.2", 4),),
        (("actuator.3", 4),),
    )

    competence = MotorPrimitive(
        primitive_id="primitive.competence",
        embodiment_fingerprint=learner.embodiment_fingerprint,
        sequence=sequence,
        samples=2,
        effect_mean=0.01,
        effect_variance=0.001,
        controllability=0.01,
        directional_consistency=0.8,
    )
    assert competence.maturity in {
        CompetenceMaturity.ESTABLISHED,
        CompetenceMaturity.ROBUST,
    }

    learner._primitives = {
        f"primitive.unverified.{index:02d}": MotorPrimitive(
            primitive_id=f"primitive.unverified.{index:02d}",
            embodiment_fingerprint=learner.embodiment_fingerprint,
            sequence=sequence,
            samples=1,
            effect_mean=1.0,
            effect_variance=0.0,
            controllability=1.0 - index * 0.001,
            directional_consistency=1.0,
        )
        for index in range(sensorimotor_module._MAX_PRIMITIVES)
    }
    learner._primitives[competence.primitive_id] = competence

    learner._enforce_primitive_bound()

    assert len(learner.primitives) == sensorimotor_module._MAX_PRIMITIVES
    assert competence.primitive_id in {primitive.primitive_id for primitive in learner.primitives}
    assert competence.primitive_id in learner.available_cognitive_primitive_ids()


def test_primitive_episode_provenance_is_ephemeral_and_independent():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-episode-provenance",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-episode-provenance", _ids(4)),
        max_concurrent=4,
    )

    _teach_repeated_sequence(learner, episodes=2)

    assert learner.last_primitive_episodes
    episode = learner.last_primitive_episodes[-1]
    assert episode.end_tick - episode.start_tick == 4
    assert episode.sample_index >= 2
    assert len(episode.evidence_blocks) >= 1
    assert episode.source == "natural"

    checkpoint = learner.checkpoint()
    assert "episodes" not in checkpoint

    learner.observe(
        tick=episode.end_tick + 1,
        body_state={"sense.a": 1.0, "sense.b": -1.0},
        motor_vector={},
        discovery_eligible=False,
    )
    assert learner.last_primitive_episodes == ()


def test_historical_primitive_requires_fresh_evidence_before_cognitive_reuse():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-historical-revalidation",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-historical-revalidation", _ids(4)),
        max_concurrent=4,
    )
    historical_id = "primitive.historical"
    sequence = (
        (("actuator.0", 5),),
        (("actuator.1", 5),),
        (("actuator.2", 5),),
        (("actuator.3", 5),),
    )

    added = learner.register_historical_primitive_candidates(
        [
            {
                "primitive_id": historical_id,
                "embodiment_fingerprint": learner.embodiment_fingerprint,
                "sequence": [
                    [["actuator.0", 5]],
                    [["actuator.1", 5]],
                    [["actuator.2", 5]],
                    [["actuator.3", 5]],
                ],
            }
        ]
    )

    assert added == 1
    assert learner.historical_primitive_candidate_ids == (historical_id,)
    assert learner.available_cognitive_primitive_ids() == ()
    assert learner.primitive_intents(historical_id) == ()

    learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=4,
        may_create=True,
        evidence_blocks=frozenset({0}),
    )
    assert learner.available_cognitive_primitive_ids() == ()

    learner._record_primitive_episode(
        sequence=sequence,
        before={"sense.x": 0.0},
        after={"sense.x": 0.1},
        end_tick=12,
        may_create=True,
        evidence_blocks=frozenset({1}),
    )

    assert learner.available_cognitive_primitive_ids() == (historical_id,)
    assert learner.historical_primitive_candidate_ids == ()
    assert learner.has_cognitive_primitive(historical_id)


def test_sensorimotor_pre_v9_checkpoint_is_not_reinterpreted_as_current_evidence():
    learner = CompetenceDevelopmentEngine(
        _ids(4),
        organism_id="org-pre-v9",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-pre-v9", _ids(4)),
    )
    payload = learner.checkpoint()
    payload["schema_version"] = 7
    payload.pop("historical_candidates")

    with pytest.raises(ValueError, match="unsupported sensorimotor checkpoint schema"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=_ids(4),
            organism_id="org-pre-v9",
        )


def test_exclusive_motor_groups_never_babble_antagonistic_channels_together():
    ids = _ids(8)
    groups = tuple((ids[index], ids[index + 1]) for index in range(0, 8, 2))
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-exclusive-units",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-exclusive-units", ids),
        exclusive_actuator_groups=groups,
    )

    seen = set()
    for tick in range(1024):
        active = {intent.actuator_id for intent in learner.motor_intents(tick)}
        seen.update(active)
        for left, right in groups:
            assert not ({left, right} <= active)
        assert len(active) <= len(groups)

    assert seen == set(ids)
    checkpoint = learner.checkpoint()
    assert checkpoint["schema_version"] == 12
    assert checkpoint["exclusive_actuator_groups"] == [list(group) for group in groups]

    restored = CompetenceDevelopmentEngine.restore(
        checkpoint,
        actuator_ids=ids,
        organism_id="org-exclusive-units",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-exclusive-units", ids),
    )
    assert restored._exclusive_actuator_groups == groups


@pytest.mark.parametrize("schema_version", [9, 10, 11])
def test_pre_current_sensorimotor_schemas_are_rejected(schema_version):
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-old-schema",
        embodiment_fingerprint="body-a",
    )
    payload = learner.checkpoint()
    payload["schema_version"] = schema_version
    with pytest.raises(ValueError, match="unsupported sensorimotor checkpoint schema"):
        CompetenceDevelopmentEngine.restore(payload, actuator_ids=ids, organism_id="org-old-schema")


def test_sensorimotor_requires_an_explicit_embodiment_fingerprint():
    ids = _ids(4)
    with pytest.raises(ValueError, match="explicit embodiment fingerprint"):
        CompetenceDevelopmentEngine(ids, organism_id="org-no-body-contract")

    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-no-restore-contract",
        embodiment_fingerprint="body-a",
    )
    with pytest.raises(ValueError, match="requires the current embodiment fingerprint"):
        CompetenceDevelopmentEngine.restore(
            learner.checkpoint(), actuator_ids=ids, organism_id="org-no-restore-contract"
        )


def test_current_checkpoint_rejects_different_body_scope():
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-v10-scope",
        embodiment_fingerprint="body-a",
    )
    payload = learner.checkpoint()

    with pytest.raises(ValueError, match="embodiment scope mismatch"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id="org-v10-scope",
            embodiment_fingerprint="body-b",
        )


def test_motor_primitive_restore_requires_its_original_body_scope():
    payload = {
        "primitive_id": "primitive.test",
        "embodiment_fingerprint": "body-a",
        "sequence": [
            [["actuator.0", 1]],
            [["actuator.1", 1]],
            [["actuator.2", 1]],
            [["actuator.3", 1]],
        ],
        "samples": 2,
        "effect_mean": 0.1,
        "effect_variance": 0.01,
        "controllability": 0.5,
        "directional_consistency": 0.75,
    }

    restored = MotorPrimitive.restore(
        payload,
        allowed_ids=set(_ids(4)),
        embodiment_fingerprint="body-a",
    )
    assert restored.embodiment_fingerprint == "body-a"

    for invalid_scope in (None, ""):
        invalid = {**payload, "embodiment_fingerprint": invalid_scope}
        with pytest.raises(ValueError, match="invalid motor primitive embodiment scope"):
            MotorPrimitive.restore(
                invalid,
                allowed_ids=set(_ids(4)),
                embodiment_fingerprint="body-b",
            )

    missing_scope = dict(payload)
    del missing_scope["embodiment_fingerprint"]
    with pytest.raises(ValueError, match="invalid motor primitive checkpoint fields"):
        MotorPrimitive.restore(
            missing_scope,
            allowed_ids=set(_ids(4)),
            embodiment_fingerprint="body-b",
        )

    with pytest.raises(ValueError, match="embodiment scope mismatch"):
        MotorPrimitive.restore(
            payload,
            allowed_ids=set(_ids(4)),
            embodiment_fingerprint="body-b",
        )


@pytest.mark.parametrize(
    "field",
    ["samples", "effect_mean", "effect_variance", "controllability", "directional_consistency"],
)
def test_motor_primitive_restore_rejects_missing_persisted_evidence(field):
    payload = {
        "primitive_id": "primitive.test",
        "embodiment_fingerprint": "body-a",
        "sequence": [
            [["actuator.0", 1]],
            [["actuator.1", 1]],
            [["actuator.2", 1]],
            [["actuator.3", 1]],
        ],
        "samples": 2,
        "effect_mean": 0.1,
        "effect_variance": 0.01,
        "controllability": 0.5,
        "directional_consistency": 0.75,
    }
    del payload[field]

    with pytest.raises(ValueError, match="invalid motor primitive checkpoint fields"):
        MotorPrimitive.restore(
            payload,
            allowed_ids=set(_ids(4)),
            embodiment_fingerprint="body-a",
        )


def test_motor_primitive_restore_rejects_unknown_checkpoint_fields():
    payload = {
        "primitive_id": "primitive.test",
        "embodiment_fingerprint": "body-a",
        "sequence": [
            [["actuator.0", 1]],
            [["actuator.1", 1]],
            [["actuator.2", 1]],
            [["actuator.3", 1]],
        ],
        "samples": 2,
        "effect_mean": 0.1,
        "effect_variance": 0.01,
        "controllability": 0.5,
        "directional_consistency": 0.75,
        "unrecognized": "must not be ignored",
    }

    with pytest.raises(ValueError, match="invalid motor primitive checkpoint fields"):
        MotorPrimitive.restore(
            payload,
            allowed_ids=set(_ids(4)),
            embodiment_fingerprint="body-a",
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_passive_effect", "incomplete sensorimotor checkpoint"),
        ("partial_running_stat", "invalid running-stat checkpoint fields"),
        ("malformed_horizon_entry", "invalid sensorimotor horizon statistic entry"),
        ("missing_horizon_count", "invalid sensorimotor horizon counts"),
        ("unknown_replay", "sensorimotor replay references an unknown primitive"),
        ("bool_inactive_replay_step", "sensorimotor replay step must be an integer"),
        ("float_inactive_replay_step", "sensorimotor replay step must be an integer"),
        ("coerced_frame_flag", "invalid sensorimotor frame discovery eligibility"),
        ("duplicate_historical_id", "duplicate historical primitive identity or sequence"),
        ("extra_episode_field", "invalid sensorimotor episode end entry"),
        ("duplicate_episode_sequence", "duplicate sensorimotor episode end sequence"),
        ("coerced_actuator_id", "invalid stored sensorimotor actuator identities"),
        ("coerced_exploration_id", "invalid sensorimotor exploration identity"),
        ("unordered_frames", "frame ticks must be strictly increasing"),
        ("empty_frame_execution_id", "invalid sensorimotor frame primitive id"),
    ],
)
def test_sensorimotor_restore_rejects_missing_or_malformed_persisted_state(mutation, error):
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-fail-closed-state",
        embodiment_fingerprint="body-a",
    )
    if mutation in {"coerced_frame_flag", "empty_frame_execution_id"}:
        learner.observe(tick=1, body_state={}, motor_vector={})
    if mutation == "unordered_frames":
        learner.observe(tick=1, body_state={}, motor_vector={})
        learner.observe(tick=2, body_state={}, motor_vector={})
    payload = learner.checkpoint()

    if mutation == "missing_passive_effect":
        del payload["passive_effect_stat"]
    elif mutation == "partial_running_stat":
        payload["passive_effect_stat"] = {"count": 0}
    elif mutation == "malformed_horizon_entry":
        payload["horizon_stats"] = [None]
    elif mutation == "missing_horizon_count":
        del payload["horizon_counts"]["1"]
    elif mutation == "unknown_replay":
        payload["replay_id"] = "not-present"
        payload["replay_step"] = 0
        payload["replay_source"] = "cognition"
    elif mutation == "bool_inactive_replay_step":
        payload["replay_step"] = False
    elif mutation == "float_inactive_replay_step":
        payload["replay_step"] = 0.0
    elif mutation == "coerced_frame_flag":
        payload["frame_history"][0]["discovery_eligible"] = "false"
    elif mutation == "duplicate_historical_id":
        candidate = {
            "primitive_id": "historical-1",
            "embodiment_fingerprint": "body-a",
            "sequence": [[(ids[0], 1)]] * 4,
        }
        payload["historical_candidates"] = [candidate, dict(candidate)]
    elif mutation == "extra_episode_field":
        payload["episode_end_ticks"] = [{"sequence": [[(ids[0], 1)]] * 4, "tick": 2, "extra": 1}]
    elif mutation == "duplicate_episode_sequence":
        row = {"sequence": [[(ids[0], 1)]] * 4, "tick": 2}
        payload["episode_end_ticks"] = [row, dict(row)]
    elif mutation == "coerced_actuator_id":
        payload["actuator_ids"][0] = 3
    elif mutation == "coerced_exploration_id":
        payload["exploration_ids"] = [3]
    elif mutation == "unordered_frames":
        payload["frame_history"][0]["tick"] = 3
    elif mutation == "empty_frame_execution_id":
        payload["frame_history"][0]["execution_primitive_id"] = ""

    with pytest.raises(ValueError, match=error):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id="org-fail-closed-state",
            embodiment_fingerprint="body-a",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"count": 0, "mean": 2.0, "m2": 0.0},
        {"count": 1, "mean": 2.0, "m2": 10.0},
    ],
)
def test_running_stat_restore_rejects_impossible_aggregates(payload):
    with pytest.raises(ValueError, match="running-stat checkpoint"):
        sensorimotor_module._RunningStat.restore(payload)


def test_sequence_distance_does_not_let_dense_support_dilute_channel_changes():
    common = tuple((f"a{i}", 5) for i in range(40))
    added = tuple((f"x{i}", 5) for i in range(10))
    left = (common,) * 4
    right = ((common + added),) * 4

    distance = CompetenceDevelopmentEngine._sequence_distance(left, right)

    assert distance >= 0.20
    assert distance > 0.10


def test_body_delta_does_not_reward_global_motion_over_strong_local_effect():
    before = {f"s{i}": 0.0 for i in range(64)}
    local = dict(before)
    local["s0"] = 1.0
    global_small = {key: 0.02 for key in before}

    local_effect = CompetenceDevelopmentEngine._body_delta(before, local)
    global_effect = CompetenceDevelopmentEngine._body_delta(before, global_small)

    assert local_effect > global_effect


def test_exclusive_group_validation_rejects_overlap_and_unknown_ids():
    ids = _ids(4)
    with pytest.raises(ValueError, match="must not overlap"):
        CompetenceDevelopmentEngine(
            ids,
            organism_id="overlap",
            embodiment_fingerprint=_test_embodiment_fingerprint("overlap", ids),
            exclusive_actuator_groups=((ids[0], ids[1]), (ids[1], ids[2])),
        )
    with pytest.raises(ValueError, match="unknown actuator"):
        CompetenceDevelopmentEngine(
            ids,
            organism_id="unknown",
            embodiment_fingerprint=_test_embodiment_fingerprint("unknown", ids),
            exclusive_actuator_groups=((ids[0], "missing"),),
        )


def test_exclusive_motor_groups_arbitrate_mixed_requests_before_evidence():
    from organism.actuation.types import MotorIntent

    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-mixed-exclusion",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-mixed-exclusion", ids),
        exclusive_actuator_groups=((ids[0], ids[1]), (ids[2], ids[3])),
    )

    constrained = learner.constrain_intents(
        (
            MotorIntent(ids[0], 0.4),
            MotorIntent(ids[1], 0.8),
            MotorIntent(ids[2], 0.5),
        )
    )

    assert [(item.actuator_id, item.activation) for item in constrained] == [
        (ids[1], 0.8),
        (ids[2], 0.5),
    ]


def test_restore_rejects_v9_primitive_that_violates_exclusive_motor_unit():
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-invalid-exclusive-restore",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-invalid-exclusive-restore", ids),
        exclusive_actuator_groups=((ids[0], ids[1]),),
    )
    payload = learner.checkpoint()
    # A v9 checkpoint claiming a simultaneous antagonistic pattern is not a
    # valid state even when the IDs themselves are known.
    sequence = [
        [[ids[0], 5], [ids[1], 4]],
        [[ids[0], 5]],
        [[ids[0], 5]],
        [[ids[0], 5]],
    ]
    payload["primitive_stats"] = [
        {
            "sequence": sequence,
            "stat": {"count": 2, "mean": 0.1, "m2": 0.0},
            "first_sample_tick": 4,
            "last_sample_tick": 16,
            "materialized_tick": 16,
            "competence_tick": 16,
            "last_evidence_blocks": [0, 2],
            "signals": {},
        }
    ]

    with pytest.raises(ValueError, match="exclusive actuator groups"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id="org-invalid-exclusive-restore",
            embodiment_fingerprint=_test_embodiment_fingerprint(
                "org-invalid-exclusive-restore", ids
            ),
        )


def test_restore_rejects_duplicate_primitive_evidence_blocks():
    ids = _ids(4)
    organism_id = "org-duplicate-evidence-blocks"
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id=organism_id,
        embodiment_fingerprint=_test_embodiment_fingerprint(organism_id, ids),
    )
    payload = learner.checkpoint()
    sequence = [[[ids[0], 5]]] * 4
    payload["primitive_stats"] = [
        {
            "sequence": sequence,
            "stat": {"count": 2, "mean": 0.1, "m2": 0.0},
            "first_sample_tick": 4,
            "last_sample_tick": 16,
            "materialized_tick": 16,
            "competence_tick": None,
            "last_evidence_blocks": [0, 0],
            "signals": {},
        }
    ]

    with pytest.raises(ValueError, match="duplicate primitive evidence block"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id=organism_id,
            embodiment_fingerprint=_test_embodiment_fingerprint(organism_id, ids),
        )


@pytest.mark.parametrize("ticks", [(4, 4), (4, 3)])
def test_observe_rejects_non_increasing_causal_ticks(ticks):
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-frame-order",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-frame-order", ids),
    )
    learner.observe(tick=ticks[0], body_state={}, motor_vector={})

    with pytest.raises(ValueError, match="strictly increasing"):
        learner.observe(tick=ticks[1], body_state={}, motor_vector={})


@pytest.mark.parametrize("execution_primitive_id", ["", 4])
def test_observe_rejects_malformed_execution_primitive_identity(execution_primitive_id):
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-frame-execution-id",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-frame-execution-id", ids),
    )

    with pytest.raises(ValueError, match="invalid sensorimotor frame primitive id"):
        learner.observe(
            tick=1,
            body_state={},
            motor_vector={},
            execution_primitive_id=execution_primitive_id,
        )


def test_restore_rejects_v9_horizon_evidence_that_violates_exclusive_motor_unit():
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-invalid-horizon-restore",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-invalid-horizon-restore", ids),
        exclusive_actuator_groups=((ids[0], ids[1]),),
    )
    payload = learner.checkpoint()
    payload["horizon_stats"] = [
        {
            "horizon": 1,
            "pattern": [[ids[0], 5], [ids[1], 4]],
            "stat": {"count": 2, "mean": 0.1, "m2": 0.0},
        }
    ]

    with pytest.raises(ValueError, match="horizon evidence violates"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id="org-invalid-horizon-restore",
            embodiment_fingerprint=_test_embodiment_fingerprint("org-invalid-horizon-restore", ids),
        )


def test_restore_rejects_current_checkpoint_missing_motor_unit_contract():
    ids = _ids(4)
    learner = CompetenceDevelopmentEngine(
        ids,
        organism_id="org-v9-missing-groups",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-v9-missing-groups", ids),
    )
    payload = learner.checkpoint()
    payload.pop("exclusive_actuator_groups")

    with pytest.raises(ValueError, match="incomplete sensorimotor checkpoint"):
        CompetenceDevelopmentEngine.restore(
            payload,
            actuator_ids=ids,
            organism_id="org-v9-missing-groups",
            embodiment_fingerprint=_test_embodiment_fingerprint("org-v9-missing-groups", ids),
        )


def _reference_sequence_distance(left, right):
    if len(left) != len(right):
        return 1.0
    step_distances = []
    for left_pattern, right_pattern in zip(left, right):
        left_map = dict(left_pattern)
        right_map = dict(right_pattern)
        left_ids = set(left_map)
        right_ids = set(right_map)
        union = left_ids | right_ids
        if not union:
            step_distances.append(0.0)
            continue
        amplitude = sum(
            abs(int(left_map.get(actuator_id, 0)) - int(right_map.get(actuator_id, 0)))
            for actuator_id in union
        ) / (7.0 * len(union))
        support = len(left_ids.symmetric_difference(right_ids)) / len(union)
        step_distances.append(max(amplitude, support))
    return sum(step_distances) / len(step_distances) if step_distances else 0.0


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((("a", 1),), (("a", 1),)),
        ((("a", 1), ("b", 7)), (("a", 7), ("b", 1))),
        ((("a", 3),), (("b", 3),)),
        ((("a", 1), ("c", 4)), (("a", 2), ("b", 5), ("c", 4))),
        ((), ()),
    ],
)
def test_pattern_distance_matches_reference_allocation_heavy_formula(left, right):
    sequence_left = (left,) * 4
    sequence_right = (right,) * 4
    assert CompetenceDevelopmentEngine._sequence_distance(
        sequence_left, sequence_right
    ) == _reference_sequence_distance(sequence_left, sequence_right)


def test_primitive_match_preserves_lexicographic_tie_break_independent_of_insertion_order():
    learner = CompetenceDevelopmentEngine(
        _ids(3),
        organism_id="org-match-tie",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-match-tie", _ids(3)),
    )
    query = ((("actuator.1", 4),),) * 4
    lower = ((("actuator.0", 4),),) * 4
    higher = ((("actuator.2", 4),),) * 4

    # Both candidates have the same distance from the query. Insert the
    # lexicographically larger one first to prove dict order is irrelevant.
    learner._primitive_stats[higher] = sensorimotor_module._RunningStat()
    learner._primitive_stats[lower] = sensorimotor_module._RunningStat()

    # Temporarily widen the acceptance threshold only for winner selection;
    # the test targets deterministic argmin semantics, not the production gate.
    original = sensorimotor_module._SEQUENCE_MATCH_THRESHOLD
    sensorimotor_module._SEQUENCE_MATCH_THRESHOLD = 1.0
    try:
        assert learner._matched_primitive_sequence(query) == lower
    finally:
        sensorimotor_module._SEQUENCE_MATCH_THRESHOLD = original


def test_primitive_match_returns_original_sequence_above_threshold():
    learner = CompetenceDevelopmentEngine(
        _ids(2),
        organism_id="org-match-threshold",
        embodiment_fingerprint=_test_embodiment_fingerprint("org-match-threshold", _ids(2)),
    )
    query = ((("actuator.0", 7),),) * 4
    candidate = ((("actuator.1", 7),),) * 4
    learner._primitive_stats[candidate] = sensorimotor_module._RunningStat()

    assert learner._matched_primitive_sequence(query) == query


def test_sequence_distance_preserves_unordered_historical_pattern_semantics():
    left = ((("actuator.2", 5), ("actuator.0", 2)),) * 4
    right = ((("actuator.0", 3), ("actuator.2", 1)),) * 4

    assert CompetenceDevelopmentEngine._sequence_distance(
        left, right
    ) == _reference_sequence_distance(left, right)


def test_sequence_distance_matches_reference_exactly_for_p2_benchmark_seed():
    rng = random.Random(127)
    ids = [f"actuator.{index:03d}" for index in range(64)]

    def sequence():
        return tuple(
            tuple((actuator_id, rng.randint(1, 7)) for actuator_id in ids if rng.random() < 0.35)
            for _ in range(4)
        )

    candidates = [sequence() for _ in range(512)]
    query = sequence()

    optimized = CompetenceDevelopmentEngine._sequence_distance(query, candidates[0])
    reference = _reference_sequence_distance(query, candidates[0])

    assert optimized == reference
