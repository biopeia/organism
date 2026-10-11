"""Conformance coverage for no-label-injection and presentation-invariance.

Pursuant to biopeia/organism#98 and ADR-0074, this test suite verifies that
external presentation, evaluator, world, body, or laboratory labels do not
become injected cognitive semantics inside Organism.
"""

from __future__ import annotations

import pytest

from organism.actuation.effects import EffectAtom, EffectRepresentation, EffectSpace
from organism.cognition.generative import (
    AgendaContaminationError,
    AgendaSource,
    GenerativeAgenda,
    GenerativeTarget,
    TargetStatus,
)
from organism.memory import EpistemicStatus, ExperienceLedger, ExperienceRecord, SourceKind


# ==============================================================================
# Boundary 1: Sensorimotor Effect Space (actuation/effects.py)
# ==============================================================================


def test_effect_space_rejects_evaluator_and_world_semantic_labels():
    """EffectSpace strictly ignores or rejects features lacking organism prefixes."""
    space = EffectSpace()

    # External evaluator/world/lab labels without organism-owned prefix
    external_changes = {
        "reward": 1.0,
        "score": 100.0,
        "evaluator_success": 1.0,
        "target_reached": 1.0,
        "world_coordinate_x": 0.5,
        "hazard_flag": -1.0,
        "environment_label_food": 0.8,
    }

    # None of these labels are admitted into the signature; observe yields None
    result = space.observe(external_changes)
    assert result is None
    assert space.pressure.admissions == 0
    assert len(space.effects) == 0


def test_effect_space_accepts_only_valid_organism_prefixes():
    """Only opaque organism prefixes ('signal.', 'latent.', 'part.', etc.) are admitted."""
    space = EffectSpace()

    mixed_changes = {
        "signal.proprioception.1": 0.5,
        "latent.cluster.0": 0.25,
        "part.actuator.right": -0.5,
        "evaluator.objective": 1.0,
        "world.target": 1.0,
    }

    representation = space.observe(mixed_changes)
    assert representation is not None
    # Verify that only the organism-owned features are present in the effect representation
    assert set(representation.feature_refs) == {
        "signal.proprioception.1",
        "latent.cluster.0",
        "part.actuator.right",
    }
    assert "evaluator.objective" not in representation.feature_refs
    assert "world.target" not in representation.feature_refs


def test_effect_representation_enforces_opaque_identity_invariants():
    """EffectRepresentation and EffectTarget reject non-organism identities."""
    with pytest.raises(ValueError, match="effect_id must be opaque and organism-owned"):
        EffectRepresentation(
            effect_id="world.food_reached",
            feature_refs=("signal.1",),
            transition_signature=(("signal.1", 1),),
            support=1,
            confidence=0.5,
        )

    with pytest.raises(ValueError, match="effect features must be organism-owned references"):
        EffectRepresentation(
            effect_id="effect.valid_1234567890abcdef",
            feature_refs=("external_evaluator_score",),
            transition_signature=(("signal.1", 1),),
            support=1,
            confidence=0.5,
        )


def test_effect_space_signature_invariance_under_arbitrary_metadata_noise():
    """Effect quantization and identity depend strictly on causal numerical values, not extraneous labels."""
    space = EffectSpace()

    # Same causal signals observed with and without external non-organism metadata
    canonical_signals = {"signal.joint_angle": 0.4, "latent.state": 0.2}
    polluted_signals = {
        "signal.joint_angle": 0.4,
        "latent.state": 0.2,
        "evaluator.ground_truth_class": 5.0,
        "world.object_category": 2.0,
    }

    rep_canonical = space.observe(canonical_signals)
    rep_polluted = space.observe(polluted_signals)

    assert rep_canonical is not None
    assert rep_polluted is not None
    assert rep_canonical.effect_id == rep_polluted.effect_id
    assert rep_canonical.feature_refs == rep_polluted.feature_refs
    assert rep_canonical.transition_signature == rep_polluted.transition_signature


# ==============================================================================
# Boundary 2: Generative Agenda (cognition/generative/agenda.py)
# ==============================================================================


def test_agenda_rejects_external_target_contamination():
    """GenerativeAgenda explicitly rejects external tasks, evaluator labels, and hidden world state."""
    agenda = GenerativeAgenda()

    with pytest.raises(AgendaContaminationError, match="external task, evaluator, or hidden world state cannot enter agenda"):
        agenda.reject_external_target("lab.study.target")

    with pytest.raises(AgendaContaminationError):
        agenda.reject_external_target({"benchmark_goal": "reach_beacon", "reward_threshold": 0.9})

    assert agenda.agenda_contamination_count == 2


def test_agenda_target_requires_endogenous_source():
    """GenerativeTarget strictly requires an endogenous AgendaSource enum, rejecting foreign source labels."""
    with pytest.raises(ValueError, match="source must be an AgendaSource"):
        GenerativeTarget(
            target_id="target-1",
            source="external_evaluator_feedback",  # type: ignore[arg-type]
            source_refs=("state-ref",),
            created_tick=0,
            uncertainty=0.5,
            persistence=0.5,
            recurrence=1,
        )


def test_agenda_checkpoint_restores_without_accepting_foreign_metadata():
    """Agenda checkpoint from_checkpoint strictly validates known organism fields and rejects injected metadata."""
    agenda = GenerativeAgenda()
    target = GenerativeTarget(
        target_id="target-1",
        source=AgendaSource.UNCERTAINTY,
        source_refs=("state-ref",),
        created_tick=0,
        uncertainty=0.5,
        persistence=0.5,
        recurrence=1,
    )
    agenda.add_target(target)
    payload = agenda.checkpoint()

    # Attempt to inject foreign evaluation or world state metadata into checkpoint
    polluted_payload = dict(payload)
    polluted_payload["world_ground_truth"] = "unlocked_door"
    with pytest.raises(ValueError, match="agenda checkpoint fields are incomplete or unknown"):
        GenerativeAgenda.from_checkpoint(polluted_payload)


# ==============================================================================
# Boundary 3: Longitudinal Experience Records (memory/experience.py, memory/ledger.py)
# ==============================================================================


def test_experience_record_content_hash_invariance_under_record_id_renaming():
    """Renaming the external/storage record_id does not alter the underlying experience content_hash."""
    record_a = ExperienceRecord(
        record_id="ep.session1.batch0.rec001",
        organism_id="org.alpha",
        tick_class=2,
        context_tokens=("signal.context.a",),
        action_token="action.primitive.1",
        outcome_tokens=("outcome.effect.1",),
        epistemic_status=EpistemicStatus.OBSERVED,
        evidence_refs=("causal.trans.1",),
        confidence_class=4,
        source_kind=SourceKind.DIRECT,
    )

    record_b = ExperienceRecord(
        record_id="ep.session2.batch99.rec999",  # Different presentation/tracking id
        organism_id="org.alpha",
        tick_class=2,
        context_tokens=("signal.context.a",),
        action_token="action.primitive.1",
        outcome_tokens=("outcome.effect.1",),
        epistemic_status=EpistemicStatus.OBSERVED,
        evidence_refs=("causal.trans.1",),
        confidence_class=4,
        source_kind=SourceKind.DIRECT,
    )

    # Identical semantic and causal content produces identical content_hash
    assert record_a.content_hash == record_b.content_hash

    # Causal content alteration changes content_hash
    record_altered = ExperienceRecord(
        record_id="ep.session1.batch0.rec001",
        organism_id="org.alpha",
        tick_class=2,
        context_tokens=("signal.context.a",),
        action_token="action.primitive.2",  # Different action token
        outcome_tokens=("outcome.effect.1",),
        epistemic_status=EpistemicStatus.OBSERVED,
        evidence_refs=("causal.trans.1",),
        confidence_class=4,
        source_kind=SourceKind.DIRECT,
    )
    assert record_a.content_hash != record_altered.content_hash


def test_experience_record_rejects_arbitrary_unvalidated_source_kinds():
    """ExperienceRecord strictly enforces internal/endogenous SourceKind, rejecting foreign evaluator labels."""
    with pytest.raises(ValueError, match="source_kind must be a SourceKind"):
        ExperienceRecord(
            record_id="rec.1",
            organism_id="org.1",
            tick_class=1,
            context_tokens=("signal.1",),
            action_token="act.1",
            outcome_tokens=("out.1",),
            epistemic_status=EpistemicStatus.OBSERVED,
            evidence_refs=("ev.1",),
            confidence_class=3,
            source_kind="EVALUATOR_REWARD_INJECTION",  # type: ignore[arg-type]
        )


def test_experience_record_rejects_unprintable_or_spaced_evaluator_strings():
    """Experience tokens enforce printable ASCII without whitespace to prevent freeform text injections."""
    with pytest.raises(ValueError, match="printable non-whitespace ASCII only"):
        ExperienceRecord(
            record_id="rec.1",
            organism_id="org.1",
            tick_class=1,
            context_tokens=("injected natural language prompt with spaces",),
            action_token="act.1",
            outcome_tokens=("out.1",),
            epistemic_status=EpistemicStatus.OBSERVED,
            evidence_refs=("ev.1",),
            confidence_class=3,
            source_kind=SourceKind.DIRECT,
        )


def test_experience_ledger_enforces_organism_boundary():
    """ExperienceLedger rejects records belonging to a foreign organism identity."""
    ledger = ExperienceLedger(organism_id="org.alpha")
    foreign_record = ExperienceRecord(
        record_id="rec.1",
        organism_id="org.beta",  # Different organism boundary
        tick_class=1,
        context_tokens=("signal.1",),
        action_token="act.1",
        outcome_tokens=("out.1",),
        epistemic_status=EpistemicStatus.OBSERVED,
        evidence_refs=("ev.1",),
        confidence_class=3,
        source_kind=SourceKind.DIRECT,
    )

    with pytest.raises(ValueError, match="experience record belongs to a different organism"):
        ledger.append(foreign_record)
