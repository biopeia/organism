from __future__ import annotations

import copy
import json
import pytest

from organism.actuation.competence import (
    CompetenceEvidence,
    CompetenceGate,
    CompetenceLibrary,
    CompetenceMaturity,
    DEFAULT_COMPETENCE_GATE,
    MotorCompetence,
)
from organism.actuation.model import (
    CausalEvidence,
    CompetenceEffectModel,
)


def _make_competence(
    competence_id: str,
    *,
    effect_id: str | None = "effect.test",
    controller_id: str = "ctrl.1",
    support: int = 4,
    failures: int = 0,
    controllability: float = 0.5,
    reproducibility: float = 0.8,
    directional_consistency: float = 0.7,
    gate: CompetenceGate = DEFAULT_COMPETENCE_GATE,
) -> MotorCompetence:
    evidence = CompetenceEvidence(
        controller_seed_ref=f"seed.{competence_id}",
        effect_evidence_refs=(f"eff_ev.{competence_id}",),
        context_evidence_refs=("ctx.1",),
        controllability_evidence_refs=(f"ctrl_ev.{competence_id}",),
        support=support,
        failures=failures,
        reproducibility=reproducibility,
        controllability=controllability,
        directional_consistency=directional_consistency,
        gate=gate,
    )
    return MotorCompetence(
        competence_id=competence_id,
        controller_id=controller_id,
        effect_id=effect_id,
        evidence=evidence,
        parent_competence_ids=("parent.0",),
        controller_strategy_ref="strategy.standard",
    )


def test_competence_library_checkpoint_roundtrip() -> None:
    library = CompetenceLibrary(max_competences=10)
    c1 = _make_competence("competence.1", support=5)
    c2 = _make_competence("competence.2", effect_id=None, support=2)
    library.add(c1)
    library.add(c2)

    persisted = json.loads(json.dumps(library.checkpoint()))
    assert persisted["schema_version"] == 1
    assert persisted["max_competences"] == 10
    assert len(persisted["competences"]) == 2

    restored = CompetenceLibrary.restore(persisted)
    assert restored._max == 10
    assert len(restored.items) == 2

    r1 = restored.get("competence.1")
    assert r1 is not None
    assert r1.competence_id == "competence.1"
    assert r1.controller_id == "ctrl.1"
    assert r1.effect_id == "effect.test"
    assert r1.parent_competence_ids == ("parent.0",)
    assert r1.controller_strategy_ref == "strategy.standard"
    assert r1.evidence.support == 5
    assert r1.evidence.reproducibility == 0.8
    assert r1.evidence.controllability == 0.5
    assert r1.maturity == CompetenceMaturity.ESTABLISHED

    r2 = restored.get("competence.2")
    assert r2 is not None
    assert r2.effect_id is None
    assert r2.evidence.support == 2

    encoded = library.to_bytes()
    assert encoded == library.to_bytes()
    restored_from_bytes = CompetenceLibrary.from_bytes(encoded)
    assert restored_from_bytes.checkpoint() == library.checkpoint()


def test_competence_library_restore_respects_custom_gate() -> None:
    custom_gate = CompetenceGate(min_support=10, min_consistency=0.9)
    library = CompetenceLibrary(max_competences=4)
    c = _make_competence("competence.custom", support=5, directional_consistency=0.7)
    library.add(c)

    persisted = library.checkpoint()
    # Restored with default gate: support=5 >= min_support(2) and consistency 0.7 >= 0.6 -> ESTABLISHED
    restored_default = CompetenceLibrary.restore(persisted)
    assert restored_default.get("competence.custom").maturity == CompetenceMaturity.ESTABLISHED

    # Restored with strict custom gate: support=5 < 10 -> EMERGING
    restored_custom = CompetenceLibrary.restore(persisted, gate=custom_gate)
    assert restored_custom.get("competence.custom").maturity == CompetenceMaturity.EMERGING


def test_competence_library_restore_rejects_malformed_payload() -> None:
    library = CompetenceLibrary(max_competences=4)
    library.add(_make_competence("c1"))
    valid = library.checkpoint()

    with pytest.raises(ValueError, match="invalid competence library checkpoint"):
        CompetenceLibrary.restore("not-a-dict")

    # Missing fields
    for field in ("schema_version", "max_competences", "competences"):
        bad = copy.deepcopy(valid)
        del bad[field]
        with pytest.raises(ValueError, match="invalid competence library checkpoint fields"):
            CompetenceLibrary.restore(bad)

    # Unsupported version
    bad = copy.deepcopy(valid)
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported competence library checkpoint version"):
        CompetenceLibrary.restore(bad)

    # Invalid max_competences
    for bad_cap in (0, -1, "10", True):
        bad = copy.deepcopy(valid)
        bad["max_competences"] = bad_cap
        with pytest.raises(ValueError, match="invalid competence library capacity"):
            CompetenceLibrary.restore(bad)

    # Unbounded entries
    bad = copy.deepcopy(valid)
    bad["max_competences"] = 1
    bad["competences"] = [_make_competence("c1").competence_id] * 2  # len=2 > max=1
    with pytest.raises(ValueError, match="invalid or unbounded competence library entries"):
        CompetenceLibrary.restore(bad)

    # Duplicate competence_id
    bad = copy.deepcopy(valid)
    bad["competences"] = [valid["competences"][0], valid["competences"][0]]
    with pytest.raises(ValueError, match="duplicate competence_id"):
        CompetenceLibrary.restore(bad)


def test_competence_effect_model_checkpoint_roundtrip() -> None:
    model = CompetenceEffectModel(max_contexts=10)
    ev1 = CausalEvidence(
        evidence_id="ev.1",
        attempt_id=None,
        intervention_signature_id=None,
        commitment_id=None,
        competence_id="comp.1",
        action_ref="act.1",
        effect_id="eff.alpha",
        context_ref="ctx.1",
        observation_tick=1,
    )
    ev2 = CausalEvidence(
        evidence_id="ev.2",
        attempt_id=None,
        intervention_signature_id=None,
        commitment_id=None,
        competence_id="comp.1",
        action_ref="act.2",
        effect_id="eff.alpha",
        context_ref="ctx.1",
        observation_tick=2,
    )
    ev3 = CausalEvidence(
        evidence_id="ev.3",
        attempt_id=None,
        intervention_signature_id=None,
        commitment_id=None,
        competence_id="comp.1",
        action_ref="act.3",
        effect_id="eff.beta",
        context_ref=None,
        observation_tick=3,
    )
    model.observe(ev1)
    model.observe(ev2)
    model.observe(ev3)

    persisted = json.loads(json.dumps(model.checkpoint()))
    assert persisted["schema_version"] == 1
    assert persisted["max_contexts"] == 10
    assert len(persisted["contexts"]) == 2

    restored = CompetenceEffectModel.restore(persisted)
    assert restored._max_contexts == 10
    assert restored.context_count == 2

    # Verify predictions match
    orig_pred = model.predict(competence_id="comp.1", context_id="ctx.1")
    rest_pred = restored.predict(competence_id="comp.1", context_id="ctx.1")
    assert orig_pred is not None
    assert rest_pred is not None
    assert rest_pred.effect_id == orig_pred.effect_id == "eff.alpha"
    assert rest_pred.support == orig_pred.support == 2
    assert rest_pred.confidence == orig_pred.confidence == 1.0

    encoded = model.to_bytes()
    assert encoded == model.to_bytes()
    restored_from_bytes = CompetenceEffectModel.from_bytes(encoded)
    assert restored_from_bytes.checkpoint() == model.checkpoint()


@pytest.mark.parametrize(
    "restore",
    (CompetenceLibrary.from_bytes, CompetenceEffectModel.from_bytes),
)
def test_model_byte_exports_reject_duplicate_json_fields(restore) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        restore(b'{"schema_version":1,"schema_version":1}')


def test_competence_effect_model_restore_rejects_malformed_payload() -> None:
    model = CompetenceEffectModel(max_contexts=4)
    model.observe(
        CausalEvidence(
            evidence_id="e1",
            attempt_id=None,
            intervention_signature_id=None,
            commitment_id=None,
            competence_id="c1",
            action_ref="act.test",
            effect_id="ef1",
            context_ref="ctx1",
            observation_tick=1,
        )
    )
    valid = model.checkpoint()

    with pytest.raises(ValueError, match="invalid competence effect model checkpoint"):
        CompetenceEffectModel.restore("invalid")

    # Missing fields
    for field in ("schema_version", "max_contexts", "contexts"):
        bad = copy.deepcopy(valid)
        del bad[field]
        with pytest.raises(ValueError, match="invalid competence effect model checkpoint fields"):
            CompetenceEffectModel.restore(bad)

    # Schema version
    bad = copy.deepcopy(valid)
    bad["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported competence effect model checkpoint version"):
        CompetenceEffectModel.restore(bad)

    # Invalid max_contexts
    bad = copy.deepcopy(valid)
    bad["max_contexts"] = 0
    with pytest.raises(ValueError, match="invalid max_contexts"):
        CompetenceEffectModel.restore(bad)

    # Unbounded contexts
    bad = copy.deepcopy(valid)
    bad["max_contexts"] = 1
    bad["contexts"] = [valid["contexts"][0], valid["contexts"][0]]
    with pytest.raises(ValueError, match="invalid or unbounded contexts list"):
        CompetenceEffectModel.restore(bad)

    # Invalid entry
    bad = copy.deepcopy(valid)
    bad["contexts"] = [{"competence_id": "c1"}]
    with pytest.raises(ValueError, match="invalid context entry"):
        CompetenceEffectModel.restore(bad)
