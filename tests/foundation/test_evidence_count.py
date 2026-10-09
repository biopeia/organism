from __future__ import annotations

import json

import pytest

from organism.foundation import EvidenceCount, SampleCount


def test_evidence_count_is_a_primitive_non_negative_sample_count() -> None:
    count = EvidenceCount(3)

    assert count == 3
    assert isinstance(count, int)
    assert repr(count) == "EvidenceCount(3)"
    assert json.dumps(count) == "3"
    assert SampleCount is EvidenceCount


@pytest.mark.parametrize("value", [-1, -10])
def test_evidence_count_rejects_negative_values(value: int) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        EvidenceCount(value)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_evidence_count_rejects_non_plain_integer_scales(value: object) -> None:
    with pytest.raises(TypeError, match="plain integer"):
        EvidenceCount(value)  # type: ignore[arg-type]


def test_evidence_count_accepts_an_existing_evidence_count() -> None:
    original = EvidenceCount(4)

    assert EvidenceCount(original) == original
