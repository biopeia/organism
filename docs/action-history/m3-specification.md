# M3: compact `ActionDomain` checkpoint with an external immutable history

**Status:** proposal for organism#183. A prototype exists and is not wired into any runtime path. This document changes no canonical format or cross-domain contract; the changes it lists are proposals that need acceptance at the integration points named below.

**Base:** organism `cedce52`, world `1e42240`.

## 1. Problem

`ActionDomain.checkpoint()` serializes every commitment and intent ever registered, and World stores one Organism checkpoint per causal tick. As a result:
- stored bytes grow O(N²) with run length;
- per-tick CPU grows O(N), because `bounded_action.py:423` runs `restore(checkpoint())` on every step.

The retained terminal records serve only the identity rules. The analysis in #183 shows which ones, and that no constant-size representation is equivalent while ids stay opaque and ticks are not monotonic.

## 2. Owner decisions (2026-10-16)

1. M3 is the target design.
2. Organism holds the semantic authority over the history. World confirms causal cuts and verifies the dependencies they need. Workspace may provide physical storage without interpreting content.
3. The id contract stays as it is: opaque ids, no tick-derived ids, no required monotonicity.
4. Historical v4 checkpoints are preserved intact. Conversion must be verifiable, reversible and non-destructive.

## 3. Responsibilities

| Domain | Owns | Must not |
|---|---|---|
| **Organism** | Which records are terminal; the history segment format and schema; registration order (`seq`); the identity rules; the in-process membership index; validation of a restored history, including the canonical `ActionDomain.restore` checks on the reconstructed state; v4 ↔ M3 conversion | do I/O for the history; depend on World internals |
| **World** | Persisting dependency bytes **before** the payload that references them and before committing the cut; recording each dependency's digest in the committed reference; verifying at commit that the declared dependencies are durable; giving Organism, on restore, a `resolve(sha256) -> bytes` that only returns bytes matching the digest | decode or interpret history content |
| **Workspace** | Physical storage tooling: offline compaction and archive of runs, including dependency blobs, byte for byte | interpret, prune or retire dependencies (D3 forbids deleting protected evidence) |

## 4. Data contracts (Organism-owned; names provisional)

### 4.1 History segment

An immutable, content-addressed blob. Its identity is the sha256 of its exact bytes, which are canonical JSON (sorted keys, no spaces, UTF-8):

```json
{
  "schema": "biopeia-organism-action-history-segment-v1",
  "organism_id": "<owner identity>",
  "previous_head": "<sha256 | null>",
  "previous_count": 123,
  "records": [
    {"kind": "commitment", "seq": 17, "record": { "...": "exact ActionCommitment.checkpoint()" }},
    {"kind": "intent",     "seq": 4,  "record": { "...": "exact ActionIntent.checkpoint()" }},
    {"kind": "closed",     "id": "<commitment_id>"}
  ]
}
```

- **One segment per step** that made at least one record terminal. Steps that create nothing terminal write no segment.
- **Records are immutable once terminal.** This was verified on 1,440,283 consecutive record pairs across all 5,769 real v4 blobs, with 0 changes.
- **`seq` is the registration order.** It restores v4 list order exactly.

### 4.2 Compact `action_domain` section

Constant size with respect to history length:

```json
{
  "schema": "biopeia-organism-action-domain-compact-v1",
  "history": {"head": "<sha256 | null>", "count": 2448},
  "next_commitment_seq": 1225, "next_intent_seq": 0,
  "domain_schema_version": 1,
  "active_commitment_id": "...", "outcome_ledger": null,
  "live_commitments": [{"seq": 1224, "record": {}}],
  "live_intents": []
}
```

- **Live records:** active commitments, and pending or active intents.
- **The outcome ledger is already bounded** (`MAX_KEYS=256` plus a window), so it stays inline.
- **Pending intents that never commit stay live.** That growth predates M3 and is out of scope (see §10).

### 4.3 Exploration checkpoint

The compact section would replace `action_domain` in a new exploration checkpoint schema (`...-exploration-checkpoint-v5`; name to be decided). `v4` remains readable forever.

## 5. Identity semantics (unchanged)

Membership answers exactly what the full history answers today:
- a commitment id or intent id is "known" if it is live **or** recorded in the history;
- a commitment id is "closed" if a `closed` record names it.

These sets are a pure function of `history.head`. Live ids are answered by the live records. The checks run in the same order as today; the intended implementation keeps the three membership sets inside `ActionDomain` beside the live records. The prototype uses an adapter instead.

## 6. Persistence protocol (one causal step)

1. Organism produces `(compact payload, segment | none)` for the step.
2. World writes the segment as a content-addressed blob (temp file, fsync, link, directory fsync), **then** the payload, **then** commits the cut. The cut's Organism reference records `dependency_head = sha256(segment)` (or the unchanged head).
3. At commit, World verifies that `dependency_head` is durable and that it equals either the previous committed head or a segment whose `previous_head` is that head.

   World reads only the segment's `previous_head` linkage field for this check. Whether that counts as "decoding" Organism bytes is a decision for the World owner. The alternative is for Organism to declare `(head, previous_head)` in the payload metadata.

## 7. Restore

1. **Cold path** (new process, older cut, other branch):
   - load the chain from `history.head` back to genesis through World's `resolve`;
   - check every digest, owner, link and count;
   - rebuild the full v4-equivalent section and run the canonical `ActionDomain.restore` on it **once**;
   - build the index.
2. **Warm path** (same process, next step): reuse the in-process index when its `ref` equals the payload's. Registration counters and live ids always come from the payload. This keeps the per-tick cost of `bounded_action.py:423` constant.

## 8. Branching and old cuts

Segments are content-addressed and link only backwards, so branches coexist:
- restoring an older cut reads exactly that cut's prefix;
- ids from an abandoned future are fresh again, as they are with full history today;
- an uncommitted step's segment becomes an unreferenced orphan, which is harmless.

The reference graph is acyclic. Segments reference predecessors only; cuts reference segments; nothing references a cut.

## 9. Failure recovery

| Failure | Outcome |
|---|---|
| Crash after the segment is durable, before payload/commit | Orphan segment; the previous cut restores; continuation branches |
| Payload or cut durable, segment missing | Restore **refuses** (`dependency unavailable`). World's commit-time check must prevent this |
| Corrupted or foreign segment, wrong count or link | Restore **refuses** |
| History consistent in digests but semantically invalid | Restore **refuses**, through the canonical `ActionDomain.restore` on the reconstruction |

## 10. v4 conversion (non-destructive)

- `v4_to_compact` validates the original through the canonical restore and emits the compact section plus segments. `compact_to_v4` is its exact inverse.
- **Real data:** all 5,769 organism v4 blobs on the workstation (45 runs) reconstruct **byte-identically** (same sha256 as World's reference), both one by one and as per-run chains.
- **Per-run chained conversion:** 1,017 MB → 50.8 MB (20×). What remains is mostly the non-history part of the Organism payload.
- Conversion produces an additional archive form only. Originals are never replaced or deleted without a D3 retention policy.

## 11. Evidence

| Check | Result |
|---|---|
| Unit tests: regressions for every M2 counterexample (warm and cold); old restore with abandoned-future ids; seeded differential with alternating warm and cold restore; missing, corrupted or foreign history; orphan and branch; canonical validation on cold restore; v4 round trip | 33 passed; full Organism suite 1,372 passed |
| Differential with the module (`validate_m3.py sweep`, 400 sequences × 150 ops, 1/10 seeded from the real tick-919 domain, alternating warm/cold restore) | **0 failing sequences in 57,630 ops**; oracle accepted 9,515 admits, 7,735 intent commits, 7,342 exploration commits, 5,744 commands, 391 closes and 297 exploration closes |
| Harness study (`differential.py`, 60k ops per model) | M3 0/400 divergent; digest-only and watermark designs non-equivalent |
| World store, per-tick write of the 4 current payloads vs M3 (compact + segment) | 15 → 22 ms at N=1,224 (one more fsync); 90 → 19 ms at N=20k |
| Cold restore walk | 26 ms at 1,224 segments; 446 ms at 20k |
| On-disk amplification of segments | 4.5× (18 MB logical → 82 MB with 4 KB blocks at 20k) |

## 12. Acceptance points before any canonical change

**World** (contract change: dependency bytes and their digest in the committed reference):

| Test | What it covers |
|---|---|
| `tests/test_owner_checkpoint_store.py` | store write and genesis |
| `tests/test_manifest.py` | `CheckpointReference` schema |
| `tests/test_durable_runtime.py` | commit and recovery |
| `tests/integration/test_durable_checkpoint_restore_slice.py` | durable restore |
| `tests/integration/test_end_to_end_causal_cycle_acceptance.py` | end-to-end causal cycle |
| `tests/integration/test_environment_dynamic_adoption_restore.py` | dynamic adoption on restore |
| `tests/integration/physical_step/*` | physical steps |
| `tests/integration/observation/test_world_post_commit_owner_publication.py` | post-commit publication |

The new reference field must be optional, so existing journals and digests still verify. That interacts with world#90.

**Organism:**
- `tests/source_unit/actuation/test_exploration_composition.py` and `test_domain.py`;
- the continuation suite;
- a new v5 checkpoint test that keeps v4 restore green.

## 13. Residual decisions

1. **World:** the dependency field in `OwnerCheckpointPayload` and `CheckpointReference`; whether World may read the segment link field or must take it from declared metadata; how the field enters the journal digest (world#90).
2. **Organism:** the v5 schema name; replacing the adapter by membership sets inside `ActionDomain`.
3. **Cold-restore cost** is O(N): 0.45 s at 20k segments. Whether periodic consolidated history snapshots are needed is open; they are not proposed.
4. **Physical layout:** 4.5× block amplification of one small file per segment. It could be packed later by Workspace's offline archive, which already handles content-addressed blobs. No change to the contract is needed.
5. **Pending intents that are never terminated** keep growing live state on the intent path. This is pre-existing, and the intent pipeline has no production caller today.
6. **Retention of dependency blobs** follows D3: nothing is deleted without a verifiable conservation and recovery policy.
