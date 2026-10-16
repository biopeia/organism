"""M3 (organism#183): compact action-domain checkpoint for exploration v5.

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

SEGMENT_SCHEMA = "biopeia-organism-action-history-segment-v1"
COMPACT_SCHEMA = "biopeia-organism-action-domain-compact-v1"
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


def _is_seq(value: object) -> bool:
    return type(value) is int and value >= 0


def _check_record(record: object) -> None:
    """Every history record has one of three exact shapes."""
    if not isinstance(record, dict):
        raise ActionHistoryError("history record must be an object")
    kind = record.get("kind")
    if kind in ("commitment", "intent"):
        key = "commitment_id" if kind == "commitment" else "intent_id"
        if (set(record) != {"kind", "seq", "record"} or not _is_seq(record["seq"])
                or not isinstance(record["record"], dict) or not isinstance(record["record"].get(key), str)
                or not record["record"][key]):
            raise ActionHistoryError(f"malformed {kind} history record")
    elif kind == "closed":
        if set(record) != {"kind", "id"} or not isinstance(record["id"], str) or not record["id"]:
            raise ActionHistoryError("malformed closed history record")
    else:
        raise ActionHistoryError(f"unknown history record kind {kind!r}")


class _NotAnAncestor(Exception):
    """Incremental verification target is not on the chain (branch or gap)."""


def load_history(ref: HistoryRef, *, organism_id: str, resolve: Resolver,
                 stop_at: HistoryRef | None = None) -> list[dict]:
    """Walk the chain from ``ref`` back to genesis and return records in order.

    Every segment is digest-checked here even if the resolver already did so;
    links and counts must agree exactly, so a prefix of a longer branch is
    returned only when ``ref`` names it.
    """
    chunks: list[list[dict]] = []
    head, count = ref.head, ref.count
    while head is not None:
        if stop_at is not None and (head, count) == (stop_at.head, stop_at.count):
            return [record for chunk in reversed(chunks) for record in chunk]
        if stop_at is not None and count <= stop_at.count:
            raise _NotAnAncestor()
        try:
            raw = resolve(head)
        except (KeyError, OSError, LookupError) as exc:
            raise ActionHistoryError(f"history dependency {head} is unavailable or invalid: {exc}") from exc
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
        for record in seg["records"]:
            _check_record(record)
        prev = HistoryRef(seg.get("previous_head"), seg.get("previous_count"))
        if prev.count + len(seg["records"]) != count:
            raise ActionHistoryError("history segment count does not match its chain position")
        chunks.append(seg["records"])
        head, count = prev.head, prev.count
    if stop_at is not None and stop_at.head is not None:
        raise _NotAnAncestor()
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
    organism_id: str | None = None
    closed_ids: set[str] = field(default_factory=set)
    recorded_commitments: set[str] = field(default_factory=set)
    recorded_intents: set[str] = field(default_factory=set)
    recorded_commitment_seqs: set[int] = field(default_factory=set)
    recorded_intent_seqs: set[int] = field(default_factory=set)
    max_commitment_seq: int = -1
    max_intent_seq: int = -1
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
    if index.organism_id is None:
        index.organism_id = organism_id
    elif index.organism_id != organism_id:
        raise ActionHistoryError("history index belongs to another organism")
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
                seq = index.seq.pop(key)
                records.append({"kind": kind, "seq": seq, "record": row})
                _record_seq(index, kind, seq)
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
def _record_seq(index: HistoryIndex, kind: str, seq: int) -> None:
    if kind == "commitment":
        index.recorded_commitment_seqs.add(seq)
        index.max_commitment_seq = max(index.max_commitment_seq, seq)
    else:
        index.recorded_intent_seqs.add(seq)
        index.max_intent_seq = max(index.max_intent_seq, seq)


def _check_seqs(payload: dict, new: list[dict], index: HistoryIndex) -> None:
    """Registration order is a set of unique seqs below the next counters.

    ``recorded_*`` are seqs already in the verified history (empty on the
    cold path, where ``new`` is the whole history); ``new`` are records not
    yet indexed; live seqs come from the payload.
    """
    for kind, recorded, recorded_max, live_key, next_key in (
        ("commitment", index.recorded_commitment_seqs, index.max_commitment_seq, "live_commitments",
         "next_commitment_seq"),
        ("intent", index.recorded_intent_seqs, index.max_intent_seq, "live_intents", "next_intent_seq"),
    ):
        live = payload.get(live_key)
        if not isinstance(live, list) or any(not isinstance(r, dict) for r in live):
            raise ActionHistoryError("malformed live records")
        seqs = [r["seq"] for r in new if r["kind"] == kind] + [r.get("seq") for r in live]
        if any(not _is_seq(q) for q in seqs) or len(set(seqs)) != len(seqs) or any(q in recorded for q in seqs):
            raise ActionHistoryError("duplicate or invalid registration sequence")
        counter = payload.get(next_key)
        if not _is_seq(counter) or any(q >= counter for q in seqs) or recorded_max >= counter:
            raise ActionHistoryError("registration counter does not exceed every used sequence")


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
    if index is not None and index.organism_id != organism_id:
        index = None                                   # a cache never crosses owners
    if index is not None and index.ref != ref:
        index = _extend(index, payload, ref, organism_id=organism_id, resolve=resolve)
    elif index is not None:
        try:                                           # same history: live seqs and counters still checked
            _check_seqs(payload, [], index)
        except ActionHistoryError:
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
        index = HistoryIndex(ref=ref, organism_id=organism_id)
        _check_seqs(payload, records, index)
        ActionDomain.restore(_full_from(payload, records))   # canonical validation, once
        for r in records:
            if r["kind"] == "commitment":
                index.recorded_commitments.add(r["record"]["commitment_id"])
                _record_seq(index, "commitment", r["seq"])
            elif r["kind"] == "intent":
                index.recorded_intents.add(r["record"]["intent_id"])
                _record_seq(index, "intent", r["seq"])
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


def _extend(index: HistoryIndex, payload: dict, ref: HistoryRef, *, organism_id: str,
            resolve: Resolver) -> HistoryIndex | None:
    """Incrementally verify a chain that extends ``index.ref``; None means "go cold".

    Only the new segments are loaded and digest/link/count-checked; the
    canonical ActionDomain validation runs on the new records with the live
    payload (a closure and its commitment are always recorded together), and
    no new id may already be retired. Any doubt falls back to the full cold
    verification, which is authoritative.
    """
    try:
        new = load_history(ref, organism_id=organism_id, resolve=resolve, stop_at=index.ref)
    except _NotAnAncestor:
        return None
    ids_c = [r["record"]["commitment_id"] for r in new if r["kind"] == "commitment"]
    ids_i = [r["record"]["intent_id"] for r in new if r["kind"] == "intent"]
    closed = [r["id"] for r in new if r["kind"] == "closed"]
    if (any(i in index.recorded_commitments for i in ids_c) or any(i in index.recorded_intents for i in ids_i)
            or any(i in index.closed_ids for i in closed)):
        return None
    live_c = {r["record"]["commitment_id"] for r in payload["live_commitments"]}
    live_i = {r["record"]["intent_id"] for r in payload["live_intents"]}
    if any(i in index.recorded_commitments for i in live_c) or any(i in index.recorded_intents for i in live_i):
        return None
    try:
        _check_seqs(payload, new, index)
        ActionDomain.restore(_full_from(payload, new))
    except ValueError:
        return None
    # All checks passed: only now advance the index.
    index.recorded_commitments.update(ids_c)
    index.recorded_intents.update(ids_i)
    index.closed_ids.update(closed)
    for r in new:
        if r["kind"] in ("commitment", "intent"):
            _record_seq(index, r["kind"], r["seq"])
    index.ref = ref
    return index


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
TRANSITION_SCHEMA = "organism-transition-v2"
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
    a checkpoint requires: None for the transition checkpoint and exploration
    v4, the history head for exploration v5.

    Raises ``ActionHistoryError`` for malformed bytes or an owner/schema
    mismatch, so World refuses the commit instead of guessing.
    """
    if schema_version == TRANSITION_SCHEMA:
        return None          # transition checkpoints declare no action-history dependency
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
