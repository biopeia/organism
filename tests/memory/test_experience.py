from dataclasses import replace

import pytest

from organism.memory import EpistemicStatus, ExperienceRecord, SourceKind


def make_record() -> ExperienceRecord:
    return ExperienceRecord(
        record_id="experience.1",
        organism_id="organism.1",
        tick_class=4,
        context_tokens=("signal.internal.1",),
        action_token="action.1",
        outcome_tokens=("outcome.1",),
        epistemic_status=EpistemicStatus.OBSERVED,
        evidence_refs=("causal.transition.1",),
        confidence_class=5,
        source_kind=SourceKind.DIRECT,
    )


def test_record_round_trips_its_canonical_payload() -> None:
    record = make_record()
    assert ExperienceRecord.restore(record.canonical_payload()) == record
    assert record.content_hash == ExperienceRecord.restore(
        record.canonical_payload()
    ).content_hash


def test_content_hash_excludes_record_identity_but_includes_experience() -> None:
    record = make_record()
    same_content = replace(record, record_id="experience.2")
    different_content = replace(record, outcome_tokens=("outcome.2",))

    assert record.content_hash == same_content.content_hash
    assert record.content_hash != different_content.content_hash


def test_model_generated_record_cannot_claim_direct_observation() -> None:
    with pytest.raises(ValueError, match="cannot be marked observed"):
        replace(make_record(), source_kind=SourceKind.MODEL)
