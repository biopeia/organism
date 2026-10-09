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
| `symbiont/src/symbiont/core/cognition/bridge.py::CognitiveBridge` and `symbiont/src/symbiont/cognition/graph.py` | `REWRITE` | `src/organism/cognition/transition.py` | Adds an explicit causal-tick state transition around graph activation. The in-memory recurrent activation state is Organism-owned; it is not yet encoded in the portable checkpoint. Readouts remain consultative and do not produce actions. |

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
- The initial identity/checkpoint boundary and copied current modules are a migration candidate, not a claim of TARGET conformance or scientific validity.

## Validation

Validation is recorded in the migration PR and CI. The source repository remains the provenance source; this repository is an independent extraction and is not yet a cut-over.
