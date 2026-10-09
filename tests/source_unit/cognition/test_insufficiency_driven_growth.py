from __future__ import annotations

from organism.cognition.structure import StructuralPlasticity
from organism.cognition.types import NodeKind


def test_explained_relation_consumes_stale_growth_evidence():
    plasticity = StructuralPlasticity(
        min_candidate_support=2,
        tentative_lifetime_ticks=8,
        cooldown_ticks=8,
    )
    for tick in range(4):
        plasticity.observe_coactivation(
            source_id="a",
            target_id="b",
            source_active=True,
            target_active=True,
            tick=tick,
            source_kind=NodeKind.SENSE,
            target_kind=NodeKind.CONCEPT,
        )

    before = plasticity.export_checkpoint()
    assert before["coactivation_counts"]

    plasticity.mark_relation_explained("a", "b")

    after = plasticity.export_checkpoint()
    assert after["coactivation_counts"] == []

    # If the relation later disappears, fresh evidence is required.
    plasticity.observe_coactivation(
        source_id="a",
        target_id="b",
        source_active=True,
        target_active=True,
        tick=10,
        source_kind=NodeKind.SENSE,
        target_kind=NodeKind.CONCEPT,
    )
    refreshed = plasticity.export_checkpoint()
    assert refreshed["coactivation_counts"][0]["count"] == 1
