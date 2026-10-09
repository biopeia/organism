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
