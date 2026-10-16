from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .experience import EpistemicStatus, ExperienceRecord, SourceKind
from .ledger import ExperienceLedger

if TYPE_CHECKING:
    from organism.cognition.sense_input import OpaqueSenseSignal, SenseProvenance

_PRINTABLE_ASCII_NO_SPACE = re.compile(r"^[\x21-\x7e]+$")


@dataclass(frozen=True, slots=True)
class ExperienceRecordingPolicy:
    """Bounded configuration for capturing organism experiences during transitions."""

    record_reception: bool = True
    record_interpretation: bool = False
    max_records: int = 128
    activation_threshold: float = 0.5
    evidence_prefix: str = "sense"

    def __post_init__(self) -> None:
        if type(self.record_reception) is not bool or type(self.record_interpretation) is not bool:
            raise ValueError("record flags must be booleans")
        if (
            isinstance(self.max_records, bool)
            or not isinstance(self.max_records, int)
            or not 16 <= self.max_records <= 8192
        ):
            raise ValueError("max_records must be an integer within [16, 8192]")
        if (
            isinstance(self.activation_threshold, bool)
            or not isinstance(self.activation_threshold, (int, float))
            or not 0.0 <= self.activation_threshold <= 1.0
        ):
            raise ValueError("activation_threshold must be a float within [0.0, 1.0]")
        if (
            not isinstance(self.evidence_prefix, str)
            or not self.evidence_prefix
            or len(self.evidence_prefix) > 32
            or not _PRINTABLE_ASCII_NO_SPACE.match(self.evidence_prefix)
        ):
            raise ValueError("evidence_prefix must be a bounded non-empty ASCII string without whitespace")


def record_sense_reception(
    organism_id: str,
    causal_tick: int,
    signals: Sequence[OpaqueSenseSignal],
    provenance: SenseProvenance,
    *,
    evidence_prefix: str = "sense",
) -> ExperienceRecord:
    """Construct an ExperienceRecord representing direct reception of sensory input.

    Semantics:
    - source_kind: DIRECT (external input through authorized causal boundary)
    - epistemic_status: OBSERVED (direct sensory receipt)
    - evidence_refs: bound to the specific causal tick and world provenance
    """
    for sig in signals:
        sig_prov = getattr(sig, "provenance", None)
        if sig_prov is not None and sig_prov != provenance:
            raise ValueError(
                f"signal channel {getattr(sig, 'channel_id', '?')!r} provenance "
                f"({sig_prov}) does not match reception provenance ({provenance})"
            )
    if not signals:
        tokens = ("signal:none",)
    else:
        tokens = tuple(
            f"ch:{sig.channel_id}:{sig.value:.4f}"
            for sig in sorted(signals, key=lambda s: s.channel_id)[:16]
        )
    return ExperienceRecord(
        record_id=f"rec-{provenance.world_id}-{provenance.causal_tick}",
        organism_id=organism_id,
        tick_class=int(causal_tick),
        context_tokens=tokens,
        action_token=None,
        outcome_tokens=(),
        epistemic_status=EpistemicStatus.OBSERVED,
        evidence_refs=(f"{evidence_prefix}:{provenance.world_id}:{provenance.causal_tick}",),
        confidence_class=7,
        source_kind=SourceKind.DIRECT,
    )


def record_cognitive_interpretation(
    organism_id: str,
    causal_tick: int,
    activations: Mapping[str, float],
    *,
    threshold: float = 0.5,
) -> ExperienceRecord | None:
    """Construct an ExperienceRecord representing internal cognitive activation / association.

    Semantics:
    - source_kind: COGNITIVE (internal state pattern elicited by cognition)
    - epistemic_status: ASSOCIATED (cognitive correlation, not direct sensory receipt)
    - evidence_refs: empty (internal hypothesis/association has no external evidence reference)
    """
    salient = [
        (node_id, val)
        for node_id, val in activations.items()
        if abs(val) >= threshold
    ]
    if not salient:
        return None
    tokens = tuple(
        f"node:{node_id}:{val:.4f}"
        for node_id, val in sorted(salient, key=lambda item: -abs(item[1]))[:16]
    )
    return ExperienceRecord(
        record_id=f"int-{causal_tick}",
        organism_id=organism_id,
        tick_class=int(causal_tick),
        context_tokens=tokens,
        action_token=None,
        outcome_tokens=(),
        epistemic_status=EpistemicStatus.ASSOCIATED,
        evidence_refs=(),
        confidence_class=4,
        source_kind=SourceKind.COGNITIVE,
    )


def record_action_outcome(
    organism_id: str,
    causal_tick: int,
    action_token: str,
    outcome_tokens: tuple[str, ...],
    evidence_ref: str,
    *,
    supported: bool = True,
    confidence_class: int = 6,
) -> ExperienceRecord:
    """Construct an ExperienceRecord representing tested knowledge from action outcome.

    Semantics:
    - source_kind: ACTION_OUTCOME (observed consequence following an executed motor intent)
    - epistemic_status: SUPPORTED or CONTRADICTED (empirical validation of expected outcome)
    - evidence_refs: bound to verified outcome receipt
    """
    if not outcome_tokens:
        raise ValueError("outcome_tokens must not be empty for action outcome knowledge")
    if not evidence_ref:
        raise ValueError("evidence_ref must not be empty for action outcome knowledge")
    return ExperienceRecord(
        record_id=f"act-{causal_tick}-{action_token}",
        organism_id=organism_id,
        tick_class=int(causal_tick),
        context_tokens=(f"action:{action_token}",),
        action_token=action_token,
        outcome_tokens=outcome_tokens,
        epistemic_status=EpistemicStatus.SUPPORTED if supported else EpistemicStatus.CONTRADICTED,
        evidence_refs=(evidence_ref,),
        confidence_class=confidence_class,
        source_kind=SourceKind.ACTION_OUTCOME,
    )


def accumulate_experience_records(
    organism_id: str,
    current_records: Sequence[ExperienceRecord],
    new_records: Sequence[ExperienceRecord],
    *,
    max_records: int = 2048,
) -> tuple[ExperienceRecord, ...]:
    """Deterministically accumulate records into an ExperienceLedger with FIFO bounds."""
    ledger = ExperienceLedger(organism_id, max_records=max_records)
    for record in current_records:
        ledger.append(record)
    for record in new_records:
        if ledger.get(record.record_id) is None:
            ledger.append(record)
    return ledger.records


__all__ = [
    "ExperienceRecordingPolicy",
    "accumulate_experience_records",
    "record_action_outcome",
    "record_cognitive_interpretation",
    "record_sense_reception",
]
