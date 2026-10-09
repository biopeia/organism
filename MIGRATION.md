# Migration record

## Provenance

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source checkout SHA: `80d372a186513035f85a769db886e505cabba52d`
- Architecture audit baseline: `cef998a55aae32ed8b2f7373079c11667b01e915`
- Migration mode: copy-first, non-destructive, reversible
- Source repository was not modified by this extraction.

## Extracted source

| Original source path | Disposition | Target path | Notes |
|---|---|---|---|
| `symbiont/src/symbiont/agency/` | `SPLIT` | `src/organism/agency/` | Copied as private agency state/capability. Imports retargeted to the standalone package. |
| `symbiont/src/symbiont/cognition/` | `MOVE` | `src/organism/cognition/` | Copied cognitive structures, learning and private-model implementation. No efficacy claim is implied. |
| `symbiont/src/symbiont/genetics/` | `MOVE` | `src/organism/genetics/` | Copied genome and inheritance mechanisms. This does not demonstrate endogenous reproduction/evolution. |
| `symbiont/src/symbiont/actuation/` except `acquisition.py` | `SPLIT` | `src/organism/actuation/` | Copied competence, action/effect and sensorimotor state. `acquisition.py` is deferred because it writes to Embodiment body-schema state. |
| `symbiont/src/symbiont/foundation/` | `SPLIT` | `src/organism/foundation/` | Copied epistemic/limit primitives used by cognition; broader foundation ownership remains open to later mapping. |
| `symbiont/src/symbiont/capacity.py` | `SPLIT` | `src/organism/capacity.py` | Copied shared capability-pressure value types used by Organism agency/cognition. |
| `symbiont/src/symbiont/provenance.py` | `SPLIT` | `src/organism/provenance.py` | Copied causal provenance primitives used by Organism action evidence. |
| `symbiont/src/symbiont/host/adaptive.py::PairAccumulator` | `SPLIT` / `REWRITE` | `src/organism/statistics.py` | Extracted only the bounded online bivariate accumulator; no host sensing/adaptive subsystem was copied. |
| `symbiont/src/symbiont/core/orchestration/runtime.py` identity handling | `REWRITE` | `src/organism/identity.py` | Requires an explicit identity value rather than implicit runtime identity generation. |
| `symbiont/src/symbiont/host/checkpoint.py` and mixed runtime checkpoint composition | `REWRITE` | `src/organism/checkpoint.py` | Opaque private-state boundary and verified content-reference closure only; no runtime persistence format is selected. |

## Tests copied

Tests under `symbiont/tests/unit/{agency,cognition,genetics,actuation}/` were copied where they exercise only the extracted source slice: 53 of 64 source test modules. Historical import paths were changed to `organism` imports. Eleven test modules are deferred because they require the mixed acquisition/body-schema path, core runtime composition, or inspect the original monorepo layout:

```text
actuation/test_acquisition_footprints.py
actuation/test_action_dimension.py
actuation/test_agency_before_competence.py
actuation/test_causal_evidence.py
actuation/test_competence_convergence.py
actuation/test_effect_atom_evidence.py
actuation/test_intervention_signature.py
actuation/test_sensorimotor_v2_architecture.py
agency/test_affordances.py
genetics/test_germline_expression_v2.py
genetics/test_genome_v2.py
```

These remain available in the source repository and are not considered deleted or invalidated.

## Known rewrites and exclusions

- Historical package imports inside the selected code/tests were rewritten to `organism`.
- `actuation/acquisition.py` remains deferred; it couples Organism acquisition evidence to Embodiment `BodySchemaEngine` writes.
- `core/orchestration/`, `core/embodiment/`, `host/`, sensory/environment logic, World, Lab and scientific assets were not copied wholesale.
- No backward-compatibility shim for the historical import path is included.
- Restoring a composed cognitive payload through an `OrganismCheckpoint` that declares external dependencies now requires an explicit dependency resolver and verifies each referenced payload digest before restoring. This validates declared references only; it does not demonstrate that current private-model artifacts are all represented in the envelope.
- The initial identity/checkpoint boundary and copied current modules are a migration candidate, not a claim of TARGET conformance or scientific validity.

## Validation

Validation is recorded in the migration PR and CI. The source repository remains the provenance source; this repository is an independent extraction and is not yet a cut-over.

## Incremental slice: private experience record

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Original path: `symbiont/src/symbiont/modeling/experience.py`
- Source blob SHA-256: `c58c07f2875df887af370728408b782906beb37ef0a4d0c18631969452f54475`
- Target path: `src/organism/memory/experience.py`
- Disposition: `SPLIT` — the standalone bounded experience value and its epistemic/source classifications are Organism-owned longitudinal-memory data; the larger modeling runtime, corpus builder, and archive/custody behavior are not included.
- Rewrite: none in the copied module; only import/package location changes for the target export.
- Tests: focused target tests cover canonical restore/hash behavior and reject model-generated records marked as direct observations. The source's private-experience pipeline test is not copied because it exercises runtime composition, actuation, and private-model training beyond this slice.
- Deferred: `modeling/ledger.py` is not included because it combines the Organism experience ledger with a historical archive/sampling class whose scientific-evidence custody boundary needs separate extraction review.
- Scope limitation: this slice does not establish scientific validity, model-training efficacy, or a complete portable Organism checkpoint.

## Incremental slice: experience records in cognitive continuation payload

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Original path: `symbiont/src/symbiont/modeling/experience.py`
- Source blob SHA-256: `c58c07f2875df887af370728408b782906beb37ef0a4d0c18631969452f54475`
- Target path: `src/organism/cognition/portable_checkpoint.py`
- Disposition: `SPLIT` — the existing `ExperienceRecord` values can be composed with the cognitive graph and recurrent state when explicitly supplied by the owner-side caller.
- Contract: each record must identify the same Organism as the transition state; duplicate record identities and foreign-Organism records are rejected. Schema version 2 explicitly contains the record list; version 1 is not accepted by this target reader. No backward-compatibility obligation is implied.
- Validation: focused portable-checkpoint tests and the complete Organism repository test suite pass (7 and 566 tests respectively); isolated wheel build/install remains to be recorded separately.
- Scope limitation: the composition includes only records passed to it. It does not discover or guarantee completeness of an experience ledger, private-model weights, training work, or all Organism longitudinal state. It is not a complete portable Organism continuation.

## Incremental slice: cognitive safety and established sensory normalization state

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Source evidence:
  - `symbiont/src/symbiont/core/cognition/bridge_checkpoint.py` (blob `31aa419a94f90661a286ddb085ad512899dd2b86`) composes `safety_state` and `sensory_normalizers` into the historical cognitive bridge checkpoint and restores them.
  - `symbiont/src/symbiont/cognition/checkpoint.py` (blob `528c8de2eb080dcb170bb5e9c7b4e4a10bd6c61b`) defines the bounded safety-state and established-normalizer codecs.
  - Supporting target files `activation.py` and `metaplasticity.py` are unchanged source copies (source blob IDs `33e7ce9f9b44dfd8b5aa03db1a85e923de7d3143` and `60cd19f498f7f50783bf924ac79c1709e4a50574`, respectively).
- Target path: `src/organism/cognition/portable_checkpoint.py`
- Disposition: `SPLIT` — compose two already extracted Organism cognitive-state values in the bounded portable cognitive payload. No mixed bridge runtime, Lab adapter, Embodiment/body state, or World persistence coordinator is copied.
- Contract: schema version 3 records optional `SafetyState` and only established `SensoryNormalizer` values using the existing codecs. Unestablished normalizers remain omitted under the source codec's existing rule. The prior schema is not accepted by the current reader; no backward-compatibility obligation is implied.
- Validation: the focused portable and graph checkpoint tests cover round-trip and malformed-state rejection. This composes only explicitly supplied values and does not prove a complete cognitive or Organism continuation, actual caller completeness, model artifact closure, or runtime continuity.

## Incremental slice: generative resident state in cognitive continuation payload

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Original paths and source blob IDs:
  - `symbiont/src/symbiont/cognition/generative/resident.py` — `9fab387ce7c6fddd2445924673ab3c2d3767255d` (last source commit `5a09866340efd6568533fc0d6c686c540976c094`)
  - `symbiont/src/symbiont/cognition/generative/persistence.py` — `ad46d0d92efd64ff23552c489d2ee1b95fd95c9d`
  - `symbiont/src/symbiont/cognition/generative/agenda.py` — `64e692b090412a781ddb8f8db558511140a4e23f`
  - `symbiont/src/symbiont/cognition/generative/scheduler.py` — `1e51824d137a57e6fe7561efe202a1153d3fc96e`
  - `symbiont/src/symbiont/cognition/generative/workspace.py` — `8f39579e670ffe4b89865bdb1413dd4bb01a3489`
- Supporting source paths were last changed at `13720935595fedabec2c376dc933629b0deae5aa` unless otherwise noted above.
- Target path: `src/organism/cognition/portable_checkpoint.py`
- Disposition: `SPLIT` — the existing Organism-owned resident codec is composed into the portable cognitive envelope only when explicitly supplied. The source resident implementation and its supporting codec modules were already extracted; this change does not duplicate or alter them.
- Rewrite: the portable envelope now records/restores the resident payload and checks that the resident identity matches the transition identity. The envelope schema is version 4; older schemas are rejected rather than implicitly migrated.
- Restore boundary: a caller may supply a `GenerativeModelRegistry`; model owners are not serialized into this checkpoint. The transient factual-evidence resolver is not serialized and must be rebound by the factual-state owner. The resident codec's own schema validation remains authoritative.
- Validation: `uv run --with pytest python -m pytest -q` — 572 passed; `uv build` produced sdist and wheel; the wheel installed into an isolated target directory, imported `organism.cognition.portable_checkpoint`, and the full suite against that isolated installation passed (572 passed); `git diff --check` passed. CI is pending PR publication.
- Scope limitation: this captures the resident's codec-declared durable internal state when passed by the caller. It does not establish that all Organism callers include it, that model-weight/dependency closure is portable, that a complete Organism continuation is captured, or that cognition is scientifically effective. No source files or scientific assets were modified.

## Incremental slice: fail-closed outcome-value ledger restore

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `cef998a55aae32ed8b2f7373079c11667b01e915`
- Original source paths:
  - `symbiont/src/symbiont/agency/value.py` — blob `24758519c5fb7b288247f0a4698542466d8483a0`; last changed at `80845c371cf6dfb6c76ab60dd005a48ad0c13367`
  - `symbiont/src/symbiont/agency/prospective.py` — blob `7c63a9c8e92df8c7e767dfede2ae5e3e095e7520`; last changed at `fb109ff05b696009fa12118727f8b57710cc9735`
  - `symbiont/src/symbiont/modeling/private_runtime.py` — caller evidence blob `dedb99e6106c48c7d49ca6f1101acf2eb69ea80d`; last changed at `e2ea9f119f1417171d37f3c7dafcac5653c54a8e`. This mixed source module was not copied by this slice.
- Target paths: `src/organism/agency/value.py`, `tests/source_unit/agency/test_value.py`
- Disposition: `REWRITE` — preserve valid v1 global and v2 contextual checkpoint semantics while rejecting malformed, duplicate, incomplete, or over-capacity payloads instead of silently skipping entries or truncating the restored state.
- Restore behavior: the codec validates the complete payload before returning a ledger. A malformed entry or capacity overflow raises `ValueError`; a caller cannot receive a partial ledger as a successful restore. Existing bounded eviction during normal `observe()` operation is unchanged.
- Validation: focused agency tests — 37 passed; full Organism source suite — 576 passed; `uv build` produced wheel and sdist; installed wheel into an isolated target directory, imported the installed module, and ran the copied ledger test module against the installed package — 24 passed; `git diff --check` passed. CI is pending PR publication.
- Scope limitation: this hardens one explicitly checkpointable Organism agency state family. It does not create a top-level Organism checkpoint, prove all callers include the ledger, establish model-registry/weight artifact closure, or claim complete longitudinal continuity. The source monorepo is unchanged.

## Incremental slice: bounded private experience ledger

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Original source path: `symbiont/src/symbiont/modeling/ledger.py` (blob `02b925fccbdf3c26556bfd48101c3c50eab8494e`)
- Source symbol: `ExperienceLedger` (the first class in the source file; the following `HistoricalExperienceArchive` is excluded).
- Target path: `src/organism/memory/ledger.py`, exported as `organism.memory.ExperienceLedger`.
- Disposition: `SPLIT` — extracted the bounded append-only recent episode ledger as Organism-owned private longitudinal memory. It preserves per-record Organism identity, stable record IDs, chronological capacity eviction, explicit checkpoint/restore, and the source schema version. Tests cover round-trip, identity isolation, duplicate rejection, oldest-record eviction, and malformed/overflow restore cases.
- Deferred: `HistoricalExperienceArchive` remains in the source file and is not copied. Its stable all-time reservoir and evidence custody semantics require separate review; this slice makes no archive, retention-completeness, scientific custody, or sampling claims.
- Scope limitation: this value is not wired into the top-level Organism runtime or a complete checkpoint. It preserves only records explicitly supplied to the ledger and does not claim complete longitudinal continuity, complete private-model artifacts, or scientific validity. The source repository remains unchanged.
