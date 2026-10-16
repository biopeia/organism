# Organism ↔ World checkpoint dependency contract: DRAFT for review

**Status:** direction selected by the owner (2026-10-16); implemented as opt-in in **biopeia/world#173** (draft, pending independent review). M3 is not connected to the runtime. The Organism side exists only as the PR #184 prototype. The World side exists only as an out-of-tree reference shim; World code is unchanged.

**Base:** organism `cedce52`, world `1e42240`.

## 1. Purpose

Let an Organism checkpoint depend on immutable, content-addressed bytes (M3 history segments) that World persists and verifies before it commits a cut, without World ever interpreting them. Existing v4 checkpoints, journals and digests remain valid byte for byte.

## 2. Roles

| Party | Responsibility |
|---|---|
| **Organism** (semantic authority) | Produces the checkpoint bytes and a **dependency declaration**. Verifies, on restore, everything about the dependency content: chain links, owner, counts, and canonical `ActionDomain` validation |
| **World** (causal authority) | Persists new dependency bytes, checks digests and declared lineage, records the dependency digest in the committed reference, refuses commits and restores that break the invariants, and provides digest-checked resolution on restore. **It never parses dependency bytes** |
| **Workspace** (physical storage) | May copy or archive dependency blobs byte for byte (offline compaction). It never interprets, prunes or retires them (D3) |

## 3. Declaration (Organism → World), per owner checkpoint

```text
DependencyDeclaration(
    head:     sha256 hex | None   # dependency this checkpoint needs (None: none yet)
    parent:   sha256 hex | None   # head of the checkpoint this one extends
    new_blob: bytes | None        # the only dependency bytes not yet durable
)
```

Organism-side invariants, enforced at construction:
- if `new_blob` is present, `sha256(new_blob) == head`;
- if `new_blob` is None, `head == parent`.

One declaration per step suffices for M3, because each new segment links to `parent` inside its own bytes. World does not need to read that link.

## 4. Proposed World API changes

These are for review and are **not implemented**.

| Item | Change |
|---|---|
| `OwnerCheckpointPayload` | optional `dependency: OwnerCheckpointDependency \| None = None` (World-side type mirroring §3) |
| Owner checkpoint store | `write_dependency(bytes) -> sha256`, using the existing content-addressed write (temp, fsync, link, directory fsync), and `resolve(sha256) -> bytes` (digest-checked). The location is derived from the digest by the store layout |
| `CheckpointReference` | optional `dependency_sha256` and `dependency_parent` (both content digests, no locator) |
| Journal descriptor `_reference_descriptor` | adds the keys `"dependency_sha256"` and `"dependency_parent"` **only when a dependency exists**. Readers accept exactly the current key set, or the current set plus `dependency_sha256` (today `durable_runtime.py:118,125` and `manifest.py:583` require the exact current set) |
| Restore composition (`causal_cycle.restore…`) | passes `resolve` to the Organism restorer for references that carry `dependency_sha256` |

## 4b. Owner-contract hook

World's `OwnerCheckpointContract` gains an optional `declared_dependency(encoded, owner_identity, schema_version) -> sha256 | None`. Organism implements it as `action_history.declared_dependency`: None for v4, the history head for v5; it raises on an owner or schema mismatch. World refuses a commit whose reference omits or contradicts that statement, and refuses a dependency from an owner whose contract lacks the hook.

## 5. Commit invariants (World)

| Id | Invariant |
|---|---|
| C1 | New dependency bytes are durable **before** the owner payload is written and before the cut is committed |
| C2 | World re-hashes: `sha256(new_blob) == head` |
| C3 | `parent` equals the `dependency_sha256` recorded for the same `(owner, owner_identity)` in the cut being extended (None if absent). Without new bytes, `head == parent`. A forked or stale writer is refused |
| C4 | `head`, if not None, resolves and is digest-valid before commit |
| C5 | The committed reference records `dependency_sha256 = head`, and the journal digest covers it |
| C6 | Nothing is overwritten or deleted. A content-addressed collision with different bytes is an error (as `FileOwnerCheckpointStore` does today) |
| C7 | Dependency bytes are only hashed and stored, never parsed |

## 6. Restore invariants

| Id | Invariant |
|---|---|
| R1 | Every `dependency_sha256` in the committed cut resolves and is digest-valid, or the restore is refused (`ManifestError`) before Organism runs |
| R2 | Organism receives `resolve(sha256) -> bytes`. Every returned value is digest-checked, and a missing blob is an error |
| R3 | Organism verifies the whole chain (links, owner, counts) and runs the canonical `ActionDomain.restore` on the reconstruction. A failure maps to `ManifestError` through the existing `(TypeError, ValueError)` handling |
| R4 | Restoring an older cut and continuing makes C3 apply relative to **that** cut. Branches coexist |
| R5 | References without `dependency_sha256` (v4) restore exactly as today |

## 7. Digests and locators (world#90)

- **`dependency_sha256` is a content reference with no locator.** World resolves it through its store layout. The dependency part of the journal digest is therefore location-independent; F10 verifies this: identical `dependency_sha256` under two roots.
- **The existing `locator` field is not changed by this contract.** References keep their absolute locator, so the root-dependence reported in world#90 remains for the payload reference itself. This contract neither decides nor blocks world#90's disposition.
- **v4 references produce byte-identical journal descriptors** (F9 compares them with World's own `_reference_descriptor`), so existing journals and digests verify unchanged.

## 8. Failure tests (reference shim on World's real `FileOwnerCheckpointStore`)

`biopeia-workspace/.local/organism-183-analysis-20261016/contract/` holds `world_dependency_shim.py` and `test_dependency_contract.py`. **12 of 12 pass.** They live outside both repositories because they import both domains.

| Test | Scenario | Expected and observed |
|---|---|---|
| F1 | crash after the dependency write, before payloads | orphan blob durable; cut not committed; previous cut restores |
| F2 | crash after payloads, before commit | cut not committed; previous cut restores |
| F3 | head declared without bytes (head ≠ parent) | commit refused |
| F4 | bytes do not match the declared head | commit refused |
| F5 | parent is not the extended cut's head (stale or forked writer) | commit refused |
| F6 | committed dependency missing | restore refused by World (R1) |
| F7 | committed dependency corrupted | restore refused by World (R1) |
| F7b | an older chain segment corrupted | World's head check passes; Organism refuses (R3) |
| F8 | restore an older cut, then continue on a branch | accepted; both branches still restore |
| F9 | v4 reference | journal descriptor identical to World's current one; bytes unchanged |
| F10 | same lineage under two roots | `dependency_sha256` and payload digest identical; only the existing `locator` differs |
| Continuity | 300 exploration steps from the real tick-919 domain, with random crashes, older-cut branches, and warm/cold restores | at every committed cut, the v5 checkpoint reconstructs exactly the v4 checkpoint of the same lineage; the checkpoint stays constant-size |

## 9. Organism-side state of PR #184

- **`ActionDomain` now holds the retired-id sets itself** (`_retired_intent_ids`, `_retired_commitment_ids`, `_retired_closed_ids`, plus `restore_live`). They are never serialized.
  - Restoring and re-checkpointing all **5,769/5,769** real v4 blobs is byte-identical; the v4 format is unchanged.
  - The full Organism suite passes: 1,374 tests.
  - Differential: 0 failures in 57,630 ops after this change.
- **v5 schema** `biopeia-organism-exploration-checkpoint-v5`, with `exploration_checkpoint_v5` (returns bytes and the declaration) and `exploration_v5_to_v4` (exact v4 equivalent).
- **Not done:** wiring into `bounded_action` or the World runtime; consolidation; hot packing; deletion of anything.

## 10. Review questions

1. **World owner:** accept §4–§6 as the World contract? In particular:
   - the optional reference field and the reader key-set relaxation;
   - one declaration per owner checkpoint;
   - C3 lineage checked by World from declared digests only.
2. **World owner (world#90):** confirm that `dependency_sha256` without a locator is acceptable regardless of the decision on the existing `locator` field.
3. **Organism owner:** accept the v5 schema name and the `restore_live` / retired-set design in `ActionDomain`.
4. **Placement of the acceptance tests** once accepted:
   - the World-side invariants (C1–C7, R1, R2, R5) as World tests against a fake opaque owner;
   - the Organism-side invariants (R3) in Organism;
   - an end-to-end continuity test in the composing repository once M3 is wired.
