import pytest

from organism.cognition.generative import GenerativeConsolidator


def test_generative_use_tracks_cross_episode_reuse_without_factual_counts():
    consolidator = GenerativeConsolidator()
    consolidator.tracker.record(
        representation_ref="r0",
        episode_id="e0",
        state_id="s0",
        model_ids=("m0",),
        hypothesis_ref="h0",
        model_disagreement=0.2,
    )
    consolidator.tracker.record(
        representation_ref="r0",
        episode_id="e1",
        state_id="s1",
        model_ids=("m1",),
        hypothesis_ref="h0",
        model_disagreement=0.6,
    )

    signal = consolidator.signal(representation_ref="r0", generative_demand=0.7)
    assert signal.recurrent_activation == 2
    assert signal.cross_episode_reuse == 1
    assert signal.independent_episode_count == 2
    # Model and episode diversity is not independent factual evidence.
    assert signal.source_diversity == 0
    assert signal.hypothesis_persistence == 2
    assert signal.model_disagreement == 0.4


def test_mature_signal_projects_only_to_external_structural_contention():
    consolidator = GenerativeConsolidator()
    for episode_id, model_id in (("e0", "m0"), ("e1", "m1")):
        consolidator.tracker.record(
            representation_ref="r0",
            episode_id=episode_id,
            state_id=f"{episode_id}-s0",
            model_ids=(model_id,),
        )
    consolidator.tracker.note_factual_sources(
        representation_ref="r0",
        source_refs=("observation-0", "observation-1"),
    )
    signal = consolidator.signal(representation_ref="r0", generative_demand=0.8)
    projection = consolidator.project_candidate(
        signal=signal,
        candidate_id="candidate-0",
        eligible_tick=12,
        mutation_payloads=("add-node:concept",),
    )
    calls = []
    assert projection is not None
    assert consolidator.submit_candidate(
        projection,
        register=lambda **kwargs: calls.append(kwargs) or True,
        mutations=(object(),),
    )
    assert calls[0]["producer_id"] == "producer.generative"
    assert calls[0]["family"] == "generative"


def test_source_diversity_grows_only_from_independent_factual_reconciliation_refs():
    consolidator = GenerativeConsolidator()
    for episode_id in ("e0", "e1", "e2"):
        consolidator.tracker.record(
            representation_ref="r0",
            episode_id=episode_id,
            state_id=f"{episode_id}.s0",
            model_ids=("m0",),
        )

    before = consolidator.signal(representation_ref="r0", generative_demand=0.7)
    assert before.recurrent_activation == 3
    assert before.independent_episode_count == 3
    assert before.source_diversity == 0

    consolidator.tracker.note_factual_sources(
        representation_ref="r0",
        source_refs=("evidence.1",),
    )
    consolidator.tracker.note_factual_sources(
        representation_ref="r0",
        source_refs=("evidence.1",),
    )
    once = consolidator.signal(representation_ref="r0", generative_demand=0.7)
    assert once.source_diversity == 1

    consolidator.tracker.note_factual_sources(
        representation_ref="r0",
        source_refs=("evidence.2",),
    )
    after = consolidator.signal(representation_ref="r0", generative_demand=0.7)
    assert after.source_diversity == 2
    assert consolidator.is_mature(after)


def test_generative_use_checkpoint_round_trip_preserves_bounded_source_diversity():
    from organism.cognition.generative.consolidation import GenerativeUseTracker

    tracker = GenerativeUseTracker()
    tracker.record(
        representation_ref="r0",
        episode_id="e0",
        state_id="s0",
        model_ids=("m0",),
        model_disagreement=0.5,
    )
    tracker.note_factual_sources(
        representation_ref="r0",
        source_refs=("evidence.1", "evidence.2"),
    )

    restored = GenerativeUseTracker.from_checkpoint(tracker.checkpoint())
    signal = restored.signal(representation_ref="r0", generative_demand=0.7)

    assert signal.recurrent_activation == 1
    assert signal.independent_episode_count == 1
    assert signal.source_diversity == 2
    assert signal.model_disagreement == 0.5


def test_generative_retention_demand_decays_with_age_without_erasing_history():
    consolidator = GenerativeConsolidator(retention_decay_steps=8)
    for tick, episode_id, evidence_ref in (
        (0, "e0", "evidence.0"),
        (1, "e1", "evidence.1"),
    ):
        consolidator.tracker.record(
            representation_ref="r0",
            episode_id=episode_id,
            state_id=f"{episode_id}.s0",
            model_ids=("m0",),
            organism_step=tick,
        )
        consolidator.tracker.note_factual_sources(
            representation_ref="r0",
            source_refs=(evidence_ref,),
        )

    fresh = consolidator.signal(
        representation_ref="r0",
        generative_demand=consolidator.demand_for_age(0),
    )
    old = consolidator.signal(
        representation_ref="r0",
        generative_demand=consolidator.demand_for_age(16),
    )

    assert consolidator.is_mature(fresh)
    assert not consolidator.is_mature(old)
    assert old.source_diversity == fresh.source_diversity == 2
    assert old.cross_episode_reuse == fresh.cross_episode_reuse == 1


def test_last_generative_use_organism_step_survives_checkpoint():
    from organism.cognition.generative.consolidation import GenerativeUseTracker

    tracker = GenerativeUseTracker()
    tracker.record(
        representation_ref="r0",
        episode_id="e0",
        state_id="s0",
        organism_step=41,
    )

    restored = GenerativeUseTracker.from_checkpoint(tracker.checkpoint())

    assert restored.last_use_organism_step("r0") == 41


def test_generative_use_checkpoint_rejects_previous_time_field_and_schema():
    import copy

    from organism.cognition.generative.consolidation import GenerativeUseTracker

    tracker = GenerativeUseTracker()
    tracker.record(
        representation_ref="r0",
        episode_id="e0",
        state_id="s0",
        organism_step=41,
    )
    checkpoint = tracker.checkpoint()

    previous_schema = copy.deepcopy(checkpoint)
    previous_schema["schema_version"] = 1
    previous_row = previous_schema["representations"][0]
    previous_row["last_use_tick"] = previous_row.pop("last_use_organism_step")
    with pytest.raises(ValueError, match="fields are incomplete or unknown"):
        GenerativeUseTracker.from_checkpoint(previous_schema)

    current_schema_with_old_key = copy.deepcopy(checkpoint)
    current_row = current_schema_with_old_key["representations"][0]
    current_row["last_use_tick"] = current_row.pop("last_use_organism_step")
    with pytest.raises(ValueError, match="fields are incomplete or unknown"):
        GenerativeUseTracker.from_checkpoint(current_schema_with_old_key)


def test_generative_use_checkpoint_preserves_representation_eviction_order():
    from organism.cognition.generative.consolidation import GenerativeUseTracker

    uninterrupted = GenerativeUseTracker()
    # Insert in reverse lexical order so a sorted checkpoint changes which
    # representation is oldest when the bounded tracker reaches capacity.
    refs = tuple(f"r{index:03d}" for index in reversed(range(256)))
    for index, ref in enumerate(refs):
        uninterrupted.record(
            representation_ref=ref,
            episode_id=f"e{index}",
            state_id=f"s{index}",
        )

    restored = GenerativeUseTracker.from_checkpoint(uninterrupted.checkpoint())
    uninterrupted.record(
        representation_ref="r-new",
        episode_id="e-new",
        state_id="s-new",
    )
    restored.record(
        representation_ref="r-new",
        episode_id="e-new",
        state_id="s-new",
    )

    assert "r255" not in uninterrupted.representation_refs
    assert "r255" not in restored.representation_refs
    assert uninterrupted.checkpoint() == restored.checkpoint()


def test_generative_use_checkpoint_rejects_incomplete_eviction_order():
    from organism.cognition.generative.consolidation import GenerativeUseTracker

    tracker = GenerativeUseTracker()
    tracker.record(representation_ref="r0", episode_id="e0", state_id="s0")
    payload = tracker.checkpoint()
    payload["eviction_order"] = []

    with pytest.raises(ValueError, match="eviction order"):
        GenerativeUseTracker.from_checkpoint(payload)
