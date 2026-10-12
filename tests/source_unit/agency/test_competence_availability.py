"""The canonical competence availability projection (Revision Coherence v1 §3.1-§3.3)."""

from __future__ import annotations

import pytest

from organism.actuation.binding import (
    BindingStatus,
    CompetenceExecutionBinding,
    CompetenceExecutionBindingRegistry,
    InvalidationReason,
    StalenessReason,
)
from organism.actuation.competence import CompetenceMaturity
from organism.actuation.context import ActuatorEvidenceContext
from organism.agency.availability import AvailabilityReason, PredictionScope, derive_availability


def _context(surface: str) -> ActuatorEvidenceContext:
    return ActuatorEvidenceContext("body.test", "episode.test", "morph.test", surface)


class _Competence:
    def __init__(self, maturity=CompetenceMaturity.ESTABLISHED, effect_id="effect.e"):
        self.competence_id = "competence.a"
        self.maturity = maturity
        self.effect_id = effect_id


def _binding(**overrides):
    fields = dict(
        competence_id="competence.a",
        surface_fingerprint="surface.s",
        applicability_fingerprint=_context("surface.s").fingerprint,
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.9,
        last_evidence_tick=5,
    )
    fields.update(overrides)
    return CompetenceExecutionBinding(**fields)


def _derive(
    binding,
    *,
    controller=True,
    suppressed=False,
    surface="surface.s",
    applicability_fingerprint=_context("surface.s").fingerprint,
    **competence,
):
    return derive_availability(
        _Competence(**competence),
        binding=binding,
        effect_known=True,
        surface_fingerprint=surface,
        applicability_fingerprint=applicability_fingerprint,
        controller_available=controller,
        suppressed=suppressed,
    )


def test_suppressed_competence_stays_predictable_and_executable_but_not_admissible():
    state = _derive(_binding(), suppressed=True)
    assert state.predictable_now and state.executable_now and state.testable_now
    assert not state.admissible_now and not state.actionable_test_now
    assert state.prediction_scope is PredictionScope.CURRENT
    assert state.reason is AvailabilityReason.EXECUTIVELY_SUPPRESSED


def test_historical_knowledge_survives_a_non_current_surface():
    stale = _binding(status=BindingStatus.STALE, status_reason=StalenessReason.SURFACE_NOT_CURRENT)
    state = _derive(stale, surface="surface.other")
    assert state.predictable_now and not state.executable_now
    assert state.prediction_scope is PredictionScope.HISTORICAL


def test_uncertain_current_scope_and_invalidated_bindings():
    controller = _binding(
        status=BindingStatus.STALE, status_reason=StalenessReason.CONTROLLER_UNAVAILABLE
    )
    assert _derive(controller, controller=False).prediction_scope is (
        PredictionScope.UNCERTAIN_CURRENT
    )
    invalid = _binding(
        status=BindingStatus.INVALIDATED,
        status_reason=InvalidationReason.CAUSAL_RELATION_REVISED,
    )
    state = _derive(invalid)
    assert not state.predictable_now and not state.executable_now
    assert state.reason is AvailabilityReason.BINDING_INVALIDATED


def test_executability_reads_live_conditions_not_a_lagging_status():
    # Recorded stale for a controller that has since become available again.
    stale = _binding(
        status=BindingStatus.STALE, status_reason=StalenessReason.CONTROLLER_UNAVAILABLE
    )
    assert _derive(stale, controller=True).executable_now
    assert not _derive(_binding(), controller=False).executable_now
    assert _derive(None).reason is AvailabilityReason.UNBOUND
    assert _derive(_binding(), maturity=CompetenceMaturity.CANDIDATE).reason is (
        AvailabilityReason.NOT_MATURE
    )


def test_binding_status_invariants():
    with pytest.raises(ValueError):
        _binding(status=BindingStatus.VALID, status_reason=StalenessReason.EVIDENCE_AGED)
    with pytest.raises(ValueError):
        _binding(status=BindingStatus.STALE)
    with pytest.raises(ValueError):
        _binding(status=BindingStatus.INVALIDATED, status_reason=StalenessReason.EVIDENCE_AGED)


def test_lifecycle_transitions_are_traced_and_evidence_revalidates():
    registry = CompetenceExecutionBindingRegistry()
    kwargs = dict(
        competence_id="competence.a",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.9,
    )
    registry.bind_from_evidence(tick=1, **kwargs)
    registry.set_status(
        "competence.a",
        BindingStatus.INVALIDATED,
        InvalidationReason.EVIDENCE_CONTRADICTED,
        tick=4,
    )
    # Only new evidence leaves INVALIDATED; a status write cannot.
    assert registry.set_status("competence.a", BindingStatus.VALID, None, tick=5) is None
    registry.bind_from_evidence(tick=9, **{**kwargs, "evidence_refs": ("ev.2",)})
    (binding,) = registry.items
    assert binding.status is BindingStatus.VALID and binding.revision == 2
    assert binding.valid_from_tick == 9 and binding.last_evidence_tick == 9
    transitions = registry.drain_transitions()
    assert [(t.previous_status, t.binding.status) for t in transitions] == [
        (BindingStatus.VALID, BindingStatus.INVALIDATED),
        (BindingStatus.INVALIDATED, BindingStatus.VALID),
    ]
    restored = CompetenceExecutionBindingRegistry.restore(registry.checkpoint())
    assert restored.items == registry.items


def test_all_pending_lifecycle_transitions_survive_checkpoint_restore():
    registry = CompetenceExecutionBindingRegistry()
    registry.bind_from_evidence(
        competence_id="competence.pending",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.9,
        tick=1,
    )
    registry.set_status(
        "competence.pending",
        BindingStatus.STALE,
        StalenessReason.SURFACE_NOT_CURRENT,
        tick=3,
        cause_refs=("surface-revision-2",),
    )
    registry.bind_from_evidence(
        competence_id="competence.pending",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.2",),
        reliability=0.9,
        controllability=0.9,
        tick=4,
    )
    registry.set_status(
        "competence.pending",
        BindingStatus.STALE,
        StalenessReason.EVIDENCE_AGED,
        tick=5,
        cause_refs=("age-window-1",),
    )
    registry.bind_from_evidence(
        competence_id="competence.pending",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.3",),
        reliability=0.9,
        controllability=0.9,
        tick=6,
    )
    registry.set_status(
        "competence.pending",
        BindingStatus.INVALIDATED,
        InvalidationReason.EVIDENCE_CONTRADICTED,
        tick=7,
        cause_refs=("contradiction-1",),
    )
    payload = registry.checkpoint()
    expected = registry.drain_transitions()
    restored = CompetenceExecutionBindingRegistry.restore(payload)

    assert len(expected) == 5
    assert restored.items == registry.items
    assert restored.drain_transitions() == expected
    assert restored.drain_transitions() == ()


def test_binding_restore_rejects_old_schema_and_missing_current_fields():
    registry = CompetenceExecutionBindingRegistry()
    old = registry.checkpoint()
    old["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported competence execution binding"):
        CompetenceExecutionBindingRegistry.restore(old)

    incomplete = registry.checkpoint()
    incomplete.pop("pending_transitions")
    with pytest.raises(ValueError, match="competence execution binding checkpoint fields"):
        CompetenceExecutionBindingRegistry.restore(incomplete)


def test_binding_restore_rejects_nonconsecutive_pending_transition_revision():
    registry = CompetenceExecutionBindingRegistry()
    registry.bind_from_evidence(
        competence_id="competence.pending",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.9,
        tick=1,
    )
    registry.set_status(
        "competence.pending",
        BindingStatus.STALE,
        StalenessReason.SURFACE_NOT_CURRENT,
        tick=2,
    )
    payload = registry.checkpoint()
    payload["pending_transitions"][0]["previous_revision"] = 5

    with pytest.raises(ValueError, match="invalid pending competence execution binding revision"):
        CompetenceExecutionBindingRegistry.restore(payload)


def test_binding_restore_requires_exact_writer_schema_and_rejects_none():
    with pytest.raises(ValueError, match="checkpoint fields"):
        CompetenceExecutionBindingRegistry.restore(None)

    payload = CompetenceExecutionBindingRegistry().checkpoint()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="checkpoint fields"):
        CompetenceExecutionBindingRegistry.restore(payload)

    payload = CompetenceExecutionBindingRegistry().checkpoint()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="unsupported competence execution binding"):
        CompetenceExecutionBindingRegistry.restore(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("competence_id", 7),
        ("surface_fingerprint", None),
        ("applicability_fingerprint", None),
        ("effect_id", ""),
        ("evidence_refs", ["ev.1", 2]),
        ("reliability", "0.9"),
        ("controllability", True),
        ("last_evidence_tick", True),
        ("valid_from_tick", "1"),
        ("status_changed_tick", False),
        ("revision", 1.0),
        ("unexpected", "ignored"),
    ],
)
def test_binding_restore_rejects_coercions_and_unknown_binding_fields(field, value):
    registry = CompetenceExecutionBindingRegistry()
    registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.8,
        tick=1,
    )
    payload = registry.checkpoint()
    payload["items"][0][field] = value

    with pytest.raises(ValueError):
        CompetenceExecutionBindingRegistry.restore(payload)


def test_binding_restore_preserves_nullable_reason_and_historical_metadata():
    registry = CompetenceExecutionBindingRegistry()
    registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.8,
        tick=1,
    )
    # These histories may outlive a bounded binding item and are written as
    # independent metadata, so restore must preserve rather than discard them.
    registry.note_unconfirmed_execution(
        "competence.historical", windows=2, commitment_id="commitment.1"
    )
    payload = registry.checkpoint()
    assert payload["items"][0]["status_reason"] is None

    restored = CompetenceExecutionBindingRegistry.restore(payload)
    assert restored.checkpoint() == payload


def test_binding_restore_preserves_repeated_unconfirmed_commitment_evidence():
    registry = CompetenceExecutionBindingRegistry()
    for _ in range(2):
        registry.note_unconfirmed_execution(
            "competence.historical", windows=1, commitment_id="commitment.1"
        )

    payload = registry.checkpoint()
    assert payload["unconfirmed_executions"]["competence.historical"] == {
        "executions": 2,
        "windows": 2,
        "commitments": ["commitment.1", "commitment.1"],
    }
    assert CompetenceExecutionBindingRegistry.restore(payload).checkpoint() == payload


def test_binding_restore_rejects_malformed_pending_transition_and_history():
    registry = CompetenceExecutionBindingRegistry()
    registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.8,
        tick=1,
    )
    registry.set_status(
        "competence.a", BindingStatus.STALE, StalenessReason.EVIDENCE_AGED,
        tick=2, cause_refs=("cause.1",),
    )
    transition_payload = registry.checkpoint()
    transition_payload["pending_transitions"][0]["extra"] = None
    with pytest.raises(ValueError, match="pending competence execution binding transition"):
        CompetenceExecutionBindingRegistry.restore(transition_payload)

    history_registry = CompetenceExecutionBindingRegistry()
    history_registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint="surface.s",
        applicability_context=_context("surface.s"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.9,
        controllability=0.8,
        tick=1,
    )
    history_registry.note_execution(
        "competence.a", matched=True, commitment_id="commitment.1", reference_size=1
    )
    history_payload = history_registry.checkpoint()
    history_payload["revision_history"]["competence.a"].pop("ref_n")
    with pytest.raises(ValueError, match="revision history"):
        CompetenceExecutionBindingRegistry.restore(history_payload)


def test_bind_from_evidence_on_different_surface_advances_revision_and_records_transition():
    registry = CompetenceExecutionBindingRegistry()
    registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint="surface.one",
        applicability_context=_context("surface.one"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.8,
        controllability=0.7,
        tick=10,
    )
    (binding1,) = registry.items
    assert binding1.revision == 0
    assert binding1.surface_fingerprint == "surface.one"
    assert binding1.valid_from_tick == 10

    # Bind to a new surface
    registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint="surface.two",
        applicability_context=_context("surface.two"),
        effect_id="effect.e",
        evidence_refs=("ev.2",),
        reliability=0.85,
        controllability=0.75,
        tick=25,
    )
    (binding2,) = registry.items
    assert binding2.revision == 1
    assert binding2.surface_fingerprint == "surface.two"
    assert binding2.valid_from_tick == 25
    assert binding2.last_evidence_tick == 25
    assert binding2.evidence_refs == ("ev.2",)

    transitions = registry.drain_transitions()
    assert len(transitions) == 1
    assert transitions[0].previous_status == BindingStatus.VALID
    assert transitions[0].previous_revision == 0
    assert transitions[0].previous_binding.evidence_refs == ("ev.1",)
    assert transitions[0].binding.revision == 1


def test_same_surface_new_body_context_starts_fresh_binding_revision():
    registry = CompetenceExecutionBindingRegistry()
    body_a = ActuatorEvidenceContext("body.a", "episode.a", "morph.a", "surface.same")
    body_b = ActuatorEvidenceContext("body.b", "episode.b", "morph.b", "surface.same")
    registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint=body_a.surface_fingerprint,
        applicability_context=body_a,
        effect_id="effect.e",
        evidence_refs=("ev.body-a",),
        reliability=0.8,
        controllability=0.7,
        tick=10,
    )

    binding = registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint=body_b.surface_fingerprint,
        applicability_context=body_b,
        effect_id="effect.e",
        evidence_refs=("ev.body-b",),
        reliability=0.85,
        controllability=0.75,
        tick=25,
    )

    assert binding.applicability_fingerprint == body_b.fingerprint
    assert binding.evidence_refs == ("ev.body-b",)
    assert binding.revision == 1
    assert binding.valid_from_tick == 25
    restored_pending = CompetenceExecutionBindingRegistry.restore(registry.checkpoint())
    transition = restored_pending.drain_transitions()[0]
    assert transition.binding == binding
    assert transition.previous_binding.applicability_fingerprint == body_a.fingerprint
    assert transition.previous_binding.evidence_refs == ("ev.body-a",)
    assert not registry.is_executable(
        _Competence(),
        surface_fingerprint=body_b.surface_fingerprint,
        applicability_fingerprint=body_a.fingerprint,
    )
    assert registry.is_executable(
        _Competence(),
        surface_fingerprint=body_b.surface_fingerprint,
        applicability_fingerprint=body_b.fingerprint,
    )
    assert not registry.is_executable(
        _Competence(),
        surface_fingerprint=body_b.surface_fingerprint,
    )
    restored = CompetenceExecutionBindingRegistry.restore(registry.checkpoint())
    assert restored.get("competence.a").applicability_fingerprint == body_b.fingerprint
    with pytest.raises(TypeError, match="applicability_context"):
        registry.bind_from_evidence(
            competence_id="competence.a",
            surface_fingerprint=body_b.surface_fingerprint,
            effect_id="effect.e",
            evidence_refs=("ev.unscoped",),
            reliability=0.8,
            controllability=0.7,
            tick=26,
        )
    assert not _derive(
        binding,
        applicability_fingerprint=body_a.fingerprint,
    ).executable_now


def test_availability_rejects_binding_from_another_applicability_context():
    body_a = ActuatorEvidenceContext("body.a", "episode.a", "morph.a", "surface.same")
    body_b = ActuatorEvidenceContext("body.b", "episode.b", "morph.b", "surface.same")
    state = _derive(
        _binding(applicability_fingerprint=body_a.fingerprint),
        applicability_fingerprint=body_b.fingerprint,
    )
    assert not state.executable_now
    assert state.reason is AvailabilityReason.APPLICABILITY_MISMATCH

    current = _derive(
        _binding(applicability_fingerprint=body_a.fingerprint),
        applicability_fingerprint=body_a.fingerprint,
    )
    assert current.executable_now


def test_mark_stale_for_surface_transitions_only_non_current_surfaces():
    registry = CompetenceExecutionBindingRegistry()
    registry.bind_from_evidence(
        competence_id="competence.a",
        surface_fingerprint="surface.old",
        applicability_context=_context("surface.old"),
        effect_id="effect.e",
        evidence_refs=("ev.1",),
        reliability=0.8,
        controllability=0.7,
        tick=10,
    )
    registry.bind_from_evidence(
        competence_id="competence.b",
        surface_fingerprint="surface.current",
        applicability_context=_context("surface.current"),
        effect_id="effect.e",
        evidence_refs=("ev.2",),
        reliability=0.9,
        controllability=0.8,
        tick=15,
    )

    invalid_ids = registry.invalid_for_surface("surface.current")
    assert invalid_ids == ("competence.a",)

    transitions = registry.mark_stale_for_surface(
        "surface.current", tick=20, cause_refs=("re-embodiment:event-1",)
    )
    assert len(transitions) == 1
    assert transitions[0].competence_id == "competence.a"
    assert transitions[0].binding.status == BindingStatus.STALE
    assert transitions[0].binding.status_reason == StalenessReason.SURFACE_NOT_CURRENT
    assert transitions[0].cause_refs == ("re-embodiment:event-1",)

    # Competence b remains valid on current surface
    binding_b = registry.get("competence.b")
    assert binding_b.status == BindingStatus.VALID
