"""PROTOTYPE for organism#183 (design M3). Not wired into any runtime path.

Compact ``ActionDomain`` checkpoint plus an external, immutable,
content-addressed history of terminal records.

* Organism owns the meaning of the history: which records are terminal, their
  registration order, and the identity rules they enforce.
* Dependency bytes are expected to be persisted and digest-verified by World,
  which hands Organism a ``resolve(sha256_hex) -> bytes`` callback on restore.
  This module never performs I/O itself.
* The canonical ``exploration-checkpoint-v4`` format and the ``ActionDomain``
  contract are unchanged. ``v4_to_compact``/``compact_to_v4`` convert losslessly
  between the full v4 ``action_domain`` section and this form.

Identity semantics are those of the current ``ActionDomain``: ids are opaque,
never reusable, and ticks are not required to be monotonic. Membership of every
id ever registered is therefore kept (in the history, and in an in-process
index), while the checkpoint itself stays constant-size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable, Mapping

from .domain import ActionDomain

SEGMENT_SCHEMA = "biopeia-organism-action-history-segment-v1-prototype"
COMPACT_SCHEMA = "biopeia-organism-action-domain-compact-v1-prototype"
_TERMINAL_INTENT = frozenset({"satisfied", "failed", "rejected", "interrupted", "invalidated"})

Resolver = Callable[[str], bytes]


class ActionHistoryError(ValueError):
    """The history referenced by a compact checkpoint is unavailable or invalid."""


def _canon(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class HistoryRef:
    """Head digest and total record count of a history chain (empty: None, 0)."""

    head: str | None
    count: int

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0 or (self.head is None) != (self.count == 0):
            raise ActionHistoryError("history reference head/count are inconsistent")


EMPTY = HistoryRef(None, 0)


def encode_segment(*, organism_id: str, previous: HistoryRef, records: list[dict]) -> bytes:
    """Bytes of one immutable segment; its identity is sha256 of these bytes."""
    if not records:
        raise ActionHistoryError("a history segment must contain records")
    return _canon({
        "schema": SEGMENT_SCHEMA,
        "organism_id": organism_id,
        "previous_head": previous.head,
        "previous_count": previous.count,
        "records": records,
    })


def load_history(ref: HistoryRef, *, organism_id: str, resolve: Resolver) -> list[dict]:
    """Walk the chain from ``ref`` back to genesis and return records in order.

    Every segment is digest-checked here even if the resolver already did so;
    links and counts must agree exactly, so a prefix of a longer branch is
    returned only when ``ref`` names it.
    """
    chunks: list[list[dict]] = []
    head, count = ref.head, ref.count
    while head is not None:
        try:
            raw = resolve(head)
        except (KeyError, OSError, LookupError) as exc:
            raise ActionHistoryError(f"history dependency {head} is unavailable") from exc
        if not isinstance(raw, bytes) or sha256(raw).hexdigest() != head:
            raise ActionHistoryError(f"history dependency {head} does not match its digest")
        try:
            seg = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ActionHistoryError("history segment is not valid JSON") from exc
        if (not isinstance(seg, dict) or seg.get("schema") != SEGMENT_SCHEMA
                or seg.get("organism_id") != organism_id or not isinstance(seg.get("records"), list)
                or not seg["records"]):
            raise ActionHistoryError("history segment schema or owner mismatch")
        prev = HistoryRef(seg.get("previous_head"), seg.get("previous_count"))
        if prev.count + len(seg["records"]) != count:
            raise ActionHistoryError("history segment count does not match its chain position")
        chunks.append(seg["records"])
        head, count = prev.head, prev.count
    return [record for chunk in reversed(chunks) for record in chunk]


# ------------------------------------------------------------ in-process index
@dataclass(slots=True)
class HistoryIndex:
    """Membership of every *terminal* id, plus registration counters.

    The sets are a pure function of ``ref`` (the content-addressed history), so
    an in-process index is valid for any payload with the same ``ref`` on any
    branch. Live ids are answered by the live records themselves. Kept in
    process between steps; rebuilt from the history on a cold restore.
    """

    ref: HistoryRef = EMPTY
    closed_ids: set[str] = field(default_factory=set)
    recorded_commitments: set[str] = field(default_factory=set)
    recorded_intents: set[str] = field(default_factory=set)
    seq: dict[tuple[str, str], int] = field(default_factory=dict)   # live records only
    next_commitment_seq: int = 0
    next_intent_seq: int = 0


def _guarded(live_domain_payload: dict, index: HistoryIndex) -> ActionDomain:
    return ActionDomain.restore_live(
        live_domain_payload,
        retired_intent_ids=index.recorded_intents,
        retired_commitment_ids=index.recorded_commitments,
        retired_closed_ids=index.closed_ids,
    )


# ---------------------------------------------------------------- checkpoint
def checkpoint_compact(domain: ActionDomain, index: HistoryIndex, *, organism_id: str
                       ) -> tuple[dict, bytes | None, HistoryIndex]:
    """Return (compact payload, new segment bytes or None, advanced index).

    The segment must be durable before the payload is committed. ``index`` is
    advanced in place and returned.
    """
    full = domain.checkpoint()
    records: list[dict] = []
    live_c: list[dict] = []
    live_i: list[dict] = []
    for kind, rows, recorded, id_key, terminal in (
        ("commitment", full["commitments"], index.recorded_commitments, "commitment_id",
         lambda r: r["status"] != "active"),
        ("intent", full["intents"], index.recorded_intents, "intent_id",
         lambda r: r["status"] in _TERMINAL_INTENT),
    ):
        for row in rows:
            ident = row[id_key]
            key = (kind, ident)
            if ident in recorded:        # terminal and already in the history
                continue
            if key not in index.seq:
                if kind == "commitment":
                    index.seq[key] = index.next_commitment_seq
                    index.next_commitment_seq += 1
                else:
                    index.seq[key] = index.next_intent_seq
                    index.next_intent_seq += 1
            if terminal(row):
                records.append({"kind": kind, "seq": index.seq.pop(key), "record": row})
                recorded.add(ident)
            else:
                (live_c if kind == "commitment" else live_i).append({"seq": index.seq[key], "record": row})
    for cid in full["closed_commitment_ids"]:
        if cid not in index.closed_ids:
            records.append({"kind": "closed", "id": cid})
            index.closed_ids.add(cid)
    segment = None
    if records:
        segment = encode_segment(organism_id=organism_id, previous=index.ref, records=records)
        index.ref = HistoryRef(sha256(segment).hexdigest(), index.ref.count + len(records))
    payload = {
        "schema": COMPACT_SCHEMA,
        "history": {"head": index.ref.head, "count": index.ref.count},
        "next_commitment_seq": index.next_commitment_seq,
        "next_intent_seq": index.next_intent_seq,
        "domain_schema_version": full["schema_version"],
        "active_commitment_id": full["active_commitment_id"],
        "outcome_ledger": full["outcome_ledger"],
        "live_commitments": live_c,
        "live_intents": live_i,
    }
    return payload, segment, index


# -------------------------------------------------------------------- restore
def _full_from(payload: dict, records: list[dict]) -> dict:
    commits = [(r["seq"], r["record"]) for r in records if r["kind"] == "commitment"]
    commits += [(r["seq"], r["record"]) for r in payload["live_commitments"]]
    intents = [(r["seq"], r["record"]) for r in records if r["kind"] == "intent"]
    intents += [(r["seq"], r["record"]) for r in payload["live_intents"]]
    for rows in (commits, intents):
        seqs = [s for s, _ in rows]
        if len(set(seqs)) != len(seqs):
            raise ActionHistoryError("duplicate registration sequence")
    return {
        "schema_version": payload["domain_schema_version"],
        "intents": [r for _, r in sorted(intents, key=lambda x: x[0])],
        "commitments": [r for _, r in sorted(commits, key=lambda x: x[0])],
        "active_commitment_id": payload["active_commitment_id"],
        "closed_commitment_ids": sorted(r["id"] for r in records if r["kind"] == "closed"),
        "outcome_ledger": payload["outcome_ledger"],
    }


def _check_payload(payload: object) -> HistoryRef:
    if not isinstance(payload, dict) or payload.get("schema") != COMPACT_SCHEMA:
        raise ActionHistoryError("not a compact action-domain payload")
    hist = payload.get("history")
    if not isinstance(hist, dict):
        raise ActionHistoryError("compact payload has no history reference")
    return HistoryRef(hist.get("head"), hist.get("count"))


def restore_compact(payload: dict, *, organism_id: str, resolve: Resolver,
                    cache: dict | None = None) -> tuple[ActionDomain, HistoryIndex]:
    """Restore a domain whose identity rules behave exactly as with full history.

    Cold path: load the referenced prefix, rebuild the full v4-equivalent
    section and run the canonical ``ActionDomain.restore`` validation on it
    once, then build the index. Warm path: ``cache`` (a dict held by the
    caller's process) keeps the last index; it is reused only when its
    ``ref`` equals the payload's reference (``checkpoint_compact`` advances it
    in place). A miss (new process, old checkpoint, other branch) is cold.
    """
    ref = _check_payload(payload)
    index = cache.get("index") if cache is not None else None
    if index is not None and index.ref != ref:
        index = None
    live_payload = {
        "schema_version": payload["domain_schema_version"],
        "intents": [r["record"] for r in sorted(payload["live_intents"], key=lambda x: x["seq"])],
        "commitments": [r["record"] for r in sorted(payload["live_commitments"], key=lambda x: x["seq"])],
        "active_commitment_id": payload["active_commitment_id"],
        "closed_commitment_ids": [],
        "outcome_ledger": payload["outcome_ledger"],
    }
    if index is None:
        records = load_history(ref, organism_id=organism_id, resolve=resolve)
        ActionDomain.restore(_full_from(payload, records))   # canonical validation, once
        index = HistoryIndex(ref=ref)
        for r in records:
            if r["kind"] == "commitment":
                index.recorded_commitments.add(r["record"]["commitment_id"])
            elif r["kind"] == "intent":
                index.recorded_intents.add(r["record"]["intent_id"])
            else:
                index.closed_ids.add(r["id"])
    # Registration order and live ids always come from the payload being restored.
    index.next_commitment_seq = payload["next_commitment_seq"]
    index.next_intent_seq = payload["next_intent_seq"]
    index.seq = {("commitment", r["record"]["commitment_id"]): r["seq"] for r in payload["live_commitments"]}
    index.seq.update({("intent", r["record"]["intent_id"]): r["seq"] for r in payload["live_intents"]})
    if cache is not None:
        cache["index"] = index
    return _guarded(live_payload, index), index


# ---------------------------------------------------------------- conversion
def v4_to_compact(action_domain: Mapping[str, Any], *, organism_id: str
                  ) -> tuple[dict, list[bytes]]:
    """Lossless, non-destructive: full v4 ``action_domain`` -> compact + segments."""
    domain = ActionDomain.restore(dict(action_domain))      # validate the original first
    payload, segment, _ = checkpoint_compact(domain, HistoryIndex(), organism_id=organism_id)
    return payload, [segment] if segment is not None else []


def compact_to_v4(payload: dict, *, organism_id: str, resolve: Resolver) -> dict:
    """Exact inverse of ``v4_to_compact`` (registration order restored by seq)."""
    ref = _check_payload(payload)
    return _full_from(payload, load_history(ref, organism_id=organism_id, resolve=resolve))


# ----------------------------------------------------------- v5 + declaration
EXPLORATION_V4_SCHEMA = "biopeia-organism-exploration-checkpoint-v4"
EXPLORATION_V5_SCHEMA = "biopeia-organism-exploration-checkpoint-v5"


@dataclass(frozen=True, slots=True)
class DependencyDeclaration:
    """What Organism declares to World for one checkpoint (contract draft).

    ``head`` is the digest the checkpoint depends on (None: no history yet);
    ``parent`` is the head of the checkpoint this one extends. ``new_blob`` is
    the only dependency not yet durable: World must persist it, check
    ``sha256(new_blob) == head`` and ``parent ==`` the extended cut's head,
    before committing. Without a new blob, ``head == parent``.
    World never interprets the bytes.
    """

    head: str | None
    parent: str | None
    new_blob: bytes | None

    def __post_init__(self) -> None:
        if self.new_blob is None:
            if self.head != self.parent:
                raise ActionHistoryError("a declaration without new bytes must keep its head")
        elif sha256(self.new_blob).hexdigest() != self.head:
            raise ActionHistoryError("declared head does not match the new dependency bytes")


def _encode_exploration(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def exploration_checkpoint_v5(v4_payload: Mapping[str, Any], domain: ActionDomain, index: HistoryIndex
                              ) -> tuple[bytes, DependencyDeclaration, HistoryIndex]:
    """Build a v5 exploration checkpoint from the fields of a v4 one.

    ``v4_payload`` supplies every non-domain field unchanged; the domain is
    taken from ``domain``. Organism-side only: World receives the bytes and
    the declaration.
    """
    parent = index.ref.head
    compact, segment, index = checkpoint_compact(domain, index, organism_id=v4_payload["organism_id"])
    payload = dict(v4_payload, schema=EXPLORATION_V5_SCHEMA, action_domain=compact)
    return _encode_exploration(payload), DependencyDeclaration(index.ref.head, parent, segment), index


def exploration_v5_to_v4(encoded: bytes, *, resolve: Resolver) -> bytes:
    """Exact v4 bytes equivalent to a v5 checkpoint (for audit and rollback)."""
    payload = json.loads(encoded)
    if payload.get("schema") != EXPLORATION_V5_SCHEMA:
        raise ActionHistoryError("not a v5 exploration checkpoint")
    domain = compact_to_v4(payload["action_domain"], organism_id=payload["organism_id"], resolve=resolve)
    return _encode_exploration(dict(payload, schema=EXPLORATION_V4_SCHEMA, action_domain=domain))


def restore_exploration_v5_domain(encoded: bytes, *, resolve: Resolver, cache: dict | None = None
                                  ) -> tuple[dict, ActionDomain, HistoryIndex]:
    """Parse a v5 checkpoint and restore its ActionDomain (other fields returned as-is)."""
    payload = json.loads(encoded)
    if payload.get("schema") != EXPLORATION_V5_SCHEMA:
        raise ActionHistoryError("not a v5 exploration checkpoint")
    domain, index = restore_compact(payload["action_domain"], organism_id=payload["organism_id"],
                                    resolve=resolve, cache=cache)
    return payload, domain, index


def declared_dependency(encoded: bytes, owner_identity: str, schema_version: str) -> str | None:
    """Organism's statement, for World's owner contract, of the dependency
    an exploration checkpoint requires: None for v4, the history head for v5.

    Raises ``ActionHistoryError`` for malformed bytes or an owner/schema
    mismatch, so World refuses the commit instead of guessing.
    """
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionHistoryError("exploration checkpoint is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema_version:
        raise ActionHistoryError("checkpoint schema does not match the declared schema version")
    if payload.get("organism_id") != owner_identity:
        raise ActionHistoryError("checkpoint owner does not match the declared owner identity")
    if schema_version == EXPLORATION_V4_SCHEMA:
        return None
    if schema_version != EXPLORATION_V5_SCHEMA:
        raise ActionHistoryError("unsupported exploration checkpoint schema")
    return _check_payload(payload.get("action_domain")).head
