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

## Incremental slice: bind an Organism controller frame to its action commitment

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Source symbols / original paths:
  - `ActionCommitment` — `symbiont/src/symbiont/actuation/commitment.py`, blob `730a7b08f7aa8bf1dade4fbf311864b97accc7e6`
  - `ControllerFrame` and `SequenceController` — `symbiont/src/symbiont/actuation/controller.py`, blob `81a9cd4ac6a7bc987cca765a40f24a45737ab8bf`
  - `MotorCommand` — `symbiont/src/symbiont/actuation/action.py`, blob `e2ebd9f5d62643e2c5922be9f7decd4c7fb72420`
- Target paths: `src/organism/actuation/command.py`, public exports in `src/organism/actuation/__init__.py`
- Disposition: `REWRITE` — the source already defines owner-owned action commitments, controller frames and provenance-bearing motor commands, but does not provide a binding operation between the frame and an active commitment. The target frame records the exact commitment ID and generation tick; binding checks active status, controller/commitment/competence identity, and exact tick equality before carrying commitment provenance into a `MotorCommand`.
- Contract: caller supplies command ID and causal tick; no host clock is read. The binding does not mutate commitment/controller state, validate or execute against a body surface, or claim that a command was accepted. Existing Embodiment remains responsible for body execution. It does not bind Modality, whose extracted v1 API is sensory transduction only, and does not claim a World-integrated causal action path.
- Tests: `tests/source_unit/actuation/test_command_binding.py` checks provenance propagation, exact explicit tick, inactive/foreign commitment and mismatched controller/competence rejection, and that the operation creates a command without executing a body action.
- Scope limitation: this is an Organism-local value binding, not a complete action lifecycle or proof that World committed the command. The source repository and scientific assets remain unchanged.

## Incremental slice: Organism experiential evidence-count primitive

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Original path: `symbiont/src/symbiont/sensory/evidence.py`
- Source blob: `d2bc5c9dee18e113ffbd5a1540dc200c5882f3a3`; last source-path change `0c87bb9012342a59564c97654f34eea17157ecee` rejects cross-scale coercion.
- Target path: `src/organism/foundation/evidence.py`; exported through `organism.foundation`.
- Disposition: `SPLIT` — move only `EvidenceCount` and its `SampleCount` alias. The value represents local Organism perceptual/cognitive estimator evidence arrivals ($T_e$); it is not Modality capability, scheduler cycles ($T_s$), physical time ($T_p$), evidence content, or factual authority. No predictive-credit state or sensory lifecycle was copied.
- Rewrite: only package placement and documentation clarify the target owner/boundary; value validation, integer behavior, repr and primitive JSON representation remain source-equivalent.
- Tests: target checks cover accepted integer/subclass values, negative and cross-scale/type rejection, alias identity, and native integer JSON encoding.
- Limitation: this extracts the shared type only. `PairwisePredictiveEvidence`, its estimator equations/checkpoint, sensor state and evidence custody remain outside this slice.

## Incremental slice: bounded attention allocation

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Original path: `symbiont/src/symbiont/core/cognition/attention.py` (blob `fc1602bfb1105f0afbf7b4bb42b399f48d0b6dad`; last path change `13720935595fedabec2c376dc933629b0deae5aa`).
- Target path: `src/organism/cognition/attention.py`
- Disposition: `SPLIT` — extracted only the candidate/allocation value types and generic `AttentionBudget` selector. The Organism-owned cognition slice accepts caller-provided candidate names, uncertainty, costs, and observation counts; it does not observe or interpret external signals.
- Behavior: preserves the source greedy uncertainty-per-ranking-cost heuristic, diminishing-return factor, name tie-break, and hard cost budget. It is explicitly not an exact knapsack solver, threat classifier, scientific result, or execution path.
- Tests: `tests/source_unit/cognition/test_attention.py` covers validation, deterministic ranking, diminishing returns, rank-only bias, budget bounds, and unknown uncertainty.
- Deferred: source's `uncertainty_from_baseline`, `bounded_uncertainty_from_baseline`, and `attend_to_host` are excluded because they depend on Host acclimation state. The source test's host-specific cases are likewise not copied. This slice does not claim that any particular caller supplies valid uncertainty/costs or that allocation is effective.

## Source inventory: historical modeling package (pinned audit)

This inventory covers every Python file and every top-level named class,
function, constant, and export list in the historical modeling package at the pinned
source revision below. It records a disposition for migration planning; a
disposition is not approval to preserve historical behavior as Biopeia TARGET.
Any retained meaning must independently satisfy Biopeia's current authority
boundaries. No historical API compatibility is required.

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Source commit: `80d372a186513035f85a769db886e505cabba52d`
- Source tree: `5ebec1bfb665fe3d5d23d4a47d1ea6900d4f7471`
- Scope: `symbiont/src/symbiont/modeling/*.py` and directly relevant tests under
  `symbiont/tests/unit/modeling/` plus
  `symbiont/tests/experimental_integrity/test_private_runtime_tick_context.py`.
- Pin verification: the source commit and all listed blobs resolve in the
  read-only source checkout. The source repository was not modified.
- Test interpretation: source tests establish only the tested historical
  behavior at this pin. They do not establish scientific efficacy, TARGET
  conformance, or a requirement that Biopeia preserve that behavior.

### Per-file and top-level symbol dispositions

`DEFER` below is provisional, not a final disposition. Each row names a closure
trigger. If that trigger establishes no required TARGET capability, close the
inventory as `DROP`; if it establishes a capability, record a fresh
owner-justified `SPLIT` or `REWRITE` and target contract. No row authorizes
copying a legacy implementation or retaining an old import/API for compatibility.

| Source file and material top-level symbols | Disposition | Target mapping / reason | Tests and consumer evidence / caveats |
|---|---|---|---|
| `__init__.py`: `__all__` and package re-exports of the symbols listed below | `REWRITE` | Re-export only target-owned contracts from their new domain modules; do not recreate the monolithic `symbiont.modeling` facade. Current Organism exports are explicit by bounded subpackage; no compatibility facade is needed. | Source tests and Lab consumers import this facade widely. Each Biopeia consumer must import its owning package or Lab adapter; changing paths is not a reason to preserve the old facade. |
| `authority.py`: `ArchitectureId`, `ModelObjective`, `TrainingBudget`, `TrainingRequest`, `TrainingAuthorization`, `ModelArtifactManifest`, `ModelTrainingAuthority` | `SPLIT` | Separate Lab-owned training governance, budget, request execution and artifact-custody contracts from any Organism-owned acquired-model phenotype/identity. No direct target counterpart currently claims the historical authority role; target generative registry state is not a training authority or artifact custodian. | Exercised with Lab trainers/studies and source `test_private_model_contract.py`; consumers include `lab/src/lab/modeling/{architectures,artifacts,trainer,study}.py` and `lab/src/lab/physics3d/private_model_training.py`. Do not transfer Lab authorization to Organism or infer scientific permission from the historical class. |
| `corpus.py`: `_DEFAULT_ALLOWED_STATES`, `CorpusManifest`, `TrainingCorpus`, `_canonical_corpus_hash`, `_split_counts`, `_admissible_record`, `build_training_corpus` | `SPLIT` | Organism may own private experience records; constructing train/validation/test datasets, curation policy and evidence custody belong to Lab's scientific apparatus. Reimplement only if a Biopeia experiment requires an explicit reviewed dataset contract. | Source `test_modeling_corpus.py`, `test_private_experience_pipeline.py`; Lab consumers include `lab/src/lab/modeling/{dataset,study}.py`, `lab/src/lab/physics3d/private_model_training.py`, and learning studies. Corpus split rules are not a universal TARGET learning protocol. |
| `culture.py`: all `MAX_*` bounds; `SocialEpistemicStatus`, `CulturalAction`, `CulturalDecisionRecord`, `CulturalPolicyConfig`, `CulturalPolicy`, `_id`, `_tokens`, `_roots`, `SocialClaim`, `ClaimGraph`, `_COMPOSITE_OPERATIONS`, `CulturalComposite`, `CompositeGraph`, `LocalAssessment`, `SocialEvidenceLedger`, `DeliveryResult`, `SocialChannel` | `DEFER` | Possible Organism-local social claims/memory must be separated from transport, delivery and experimental evaluation. No social/cultural target contract is present in this repository; do not import the historical taxonomy or message transport as authority. | Source `test_private_model_contract.py` and communication/cultural studies; consumers include `lab/src/lab/integration/integrated_habitat.py` and `lab/src/lab/reproduction/runtime.py`. **Closure trigger:** approve the Biopeia social-claim/epistemic contract and owner split (Organism state vs World coordination vs Lab apparatus), then disposition each symbol; absent justified capability, close as `DROP`. |
| `episodic.py`: `__all__`, `EpisodicMemoryError`, `_valid_token`, `_bounded_unique`, `_jaccard`, `_is_causal_observation`, `_coarse_effect_features`, `EpisodicProjection`, `EpisodeStep`, `ExperienceEpisode`, `EpisodeMatch`, `ConsolidatedContingency`, `EpisodicPrediction`, `CognitiveReplay`, `EpisodicMemoryMetrics`, `_PendingObservation`, `EpisodicExperienceMemory` | `DEFER` | Current target has `memory/ExperienceRecord`, `memory/ExperienceLedger` and cognitive generative episode/replay modules, but no demonstrated mapping to this sparse-episodic index, matching heuristic or private-model coupling. Do not alias them by name. | Source `test_episodic.py`, `test_episodic_projection_trusted.py`; consumers include `lab/src/lab/studies/learning/episodic_memory_utility.py`, `generative_replay_utility.py`, and `PrivateModelOrganismRuntime`. **Closure trigger:** establish whether any episodic phenomenon is required in target independently of the historical implementation; map tested state/identity/provenance to target cognition and Lab evidence, or close as `DROP`. |
| `experience.py`: `_MAX_CONTEXT_TOKENS`, `_MAX_OUTCOME_TOKENS`, `_MAX_EVIDENCE_REFS`, `_MAX_TOKEN_LENGTH`, `_MAX_RECORD_ID_LENGTH`, `_MAX_ORGANISM_ID_LENGTH`, `_PRINTABLE_ASCII_NO_SPACE`, `EpistemicStatus`, `SourceKind`, `_validate_identifier`, `_validate_tokens`, `ExperienceRecord` | `SPLIT` | Bounded Organism-owned experience record is already extracted to `src/organism/memory/experience.py`; records can also be explicitly composed in `src/organism/cognition/portable_checkpoint.py`. This is a semantic slice, not the source model-training pipeline. | Target `tests/memory/test_experience.py` and portable-checkpoint tests cover the extracted record and selected identity/restore invariants. Historical source `test_private_experience_pipeline.py` is intentionally not copied: it exercises runtime, curation and training beyond this value slice. No completeness of an Organism memory/checkpoint is claimed. |
| `gateway.py`: `TokenPrediction`, `ModelInferenceResult`, `PrivateModelGateway`, `PrivateModelBridge` | `SPLIT` | Separate implementation-side inference capability from Organism's epistemic handling of a model prediction. Target `cognition/generative/adapters.py`, `hypothesis.py` and `reconciliation.py` are related target-owned mechanisms, not drop-in equivalents to the old bridge. Keep model execution/artifacts outside semantic ownership unless a new contract explicitly says otherwise. | Source `test_private_model_contract.py`, `test_private_model_pipeline` coverage in Lab, and `test_modeling_tokenizer.py`; consumers include `lab/src/lab/modeling/gateway.py` and Physics3D training/runtime composition. No target runtime bridge or inference equivalence has been demonstrated. |
| `ledger.py`: `ExperienceLedger`, `HistoricalExperienceArchive`, `__all__` | `SPLIT` | `ExperienceLedger` is already split into `src/organism/memory/ledger.py` and is Organism-owned bounded recent private memory. `HistoricalExperienceArchive` remains deferred: the archive's stable reservoir/sampling and scientific-evidence custody are not the same contract as the recent ledger. | Target `tests/memory/test_ledger.py` covers bounded ledger identity, eviction and restore. Source archive consumers include `lab/src/lab/studies/learning/private_model_learnability.py` and its integration test. **Archive closure trigger:** determine evidence custody, retention, sampling and reproducibility owner; then independently justify `REWRITE` in Lab or an owner-approved scientific archive, otherwise `DROP`. |
| `private_runtime.py`: `_evidence_ref`, `_opaque_class`, `_PrivateFrame`, `_MAX_CAPTURED_SENSES`, `_MAX_TEMPORAL_OUTCOMES`, `PrivateModelOrganismRuntime` | `DEFER` | The historical subclass composes private model training, host percepts, episodic capture and the monolithic runtime. Target instead has separately owned cognitive mechanisms and an explicitly partial portable cognitive checkpoint. Do not port this runtime or imply it is a complete continuation. | Source `test_private_experience_pipeline.py`, `test_private_runtime_tick_context.py` and re-embodiment/training Lab tests; consumers include `lab/src/lab/physics3d/runtime.py`, `lab/src/lab/cli/observed_resident.py`, and several learning studies. **Closure trigger:** the first Biopeia-owned runtime integration must state whether private-model execution is required, how owner state/artifact references are joined, and what checkpoint/restart proves; decompose then, or close as `DROP` if no TARGET caller exists. |
| `proposals.py`: `_MAX_TOKEN`, `_MAX_MODEL_ID`, `_token`, `_model_id`, `ModelPredictionProposal`, `ModelHypothesisProposal`, `confidence_class` | `REWRITE` | The epistemic distinction that a prediction/hypothesis is not evidence is represented through target generative hypothesis/reconciliation mechanisms. Reuse the distinction, not the historical proposal schema or tokens; target identity, provenance and authority must follow the approved semantic contracts. | Source `test_private_model_contract.py` and runtime tests; consumers are the historical private runtime/bridge. Target generative hypothesis and reconciliation tests exercise target-specific records; they do not establish one-to-one equivalence. |
| `registry.py`: `ModelState`, `_ALLOWED_TRANSITIONS`, `ModelRecord`, `ModelRegistry` | `SPLIT` | Target `src/organism/cognition/generative/registry.py` and `resident.py` own selected generative-model registry/resident state. Historical training lifecycle, artifact hashes, evaluations and ancestry fields need separation from semantic Organism state and Lab artifact/evaluation custody; do not carry the old transition graph by default. | Source `test_private_model_contract.py`, Lab promotion/training/evaluation tests; consumers include `lab/src/lab/physics3d/private_model_training.py` and `lab/src/lab/modeling`. Target generative-registry tests cover the target model, not artifact portability or full private-model continuity. |
| `responsibility.py`: `ResponsibilitySnapshot`, `TemporalResponsibilityTracker` | `SPLIT` | Comparative mechanism responsibility/loss tracking is an experimental analysis/evaluation capability, not authoritative Organism state by default. If retained, place the evaluated metric and provenance in Lab; do not add it to Organism's private state without a distinct causal need. | Source `test_temporal_responsibility.py`; source consumers include `lab/src/lab/modeling/study.py`. Target Organism has no counterpart. Lab retention must preserve the exact comparison scope and not elevate metric values to scientific conclusions without protocol evidence. |
| `runtime.py`: `_PRIVATE_LEARNING_MIN_BOOTSTRAP_TRANSITIONS`, `_PRIVATE_LEARNING_MIN_NEW_TRANSITIONS`, `_PRIVATE_LEARNING_FORCE_NEW_TRANSITIONS`, `_PRIVATE_LEARNING_VALIDATION_WINDOW`, `_PRIVATE_LEARNING_MIN_VALIDATIONS`, `_PRIVATE_LEARNING_CONTRADICTION_TRIGGER`, `AutonomousTrainingPlan`, `private_causal_records`, `ModeledOrganismRuntime` | `REWRITE` | The historical subclass/API composes model training, records, social channels and runtime behavior. Replace any necessary causal interaction with explicit Organism-owned state and Lab-requested training/intervention contracts; no historical runtime class or behavioral compatibility should be migrated. | Source `test_sequence_substrate.py`, `test_private_runtime_tick_context.py`, Lab integration/reproduction and private-model studies. Numerous Lab consumers import `ModeledOrganismRuntime`; each needs explicit target composition, not a compatibility alias. This row does not authorize such implementation in this documentation-only change. |
| `sequences.py`: `MAX_SEQUENCE_LENGTH`, `MAX_SEQUENCES`, `_id`, `SymbolSequence`, `SequenceMessage`, `SequenceDecisionRecord`, `SequenceAssociation`, `SequenceGroundingLedger`, `SequenceChannel`, `choose_sequence`, `__all__` | `DEFER` | Potential receiver-local sequence learning is distinct from a Lab-controlled channel/delivery mechanism. No target contract presently maps sequence identity, sender/receiver, causal time, provenance or delivery ownership. | Source `test_sequence_substrate.py`, `test_sequence_characterization_ceiling.py`; consumers include `lab/src/lab/integration/integrated_habitat.py`, `lab/src/lab/reproduction/runtime.py` and structured-communication studies. **Closure trigger:** approve target communication semantics and establish whether the capability belongs in Organism, World coordination or Lab; then disposition symbols individually, otherwise `DROP`. |
| `symbols.py`: `MAX_SYMBOLS`, `MAX_ID`, `MAX_ASSOCIATIONS`, `MAX_HISTORY`, `MAX_EMISSIONS_PER_TICK`, `_id`, `build_opaque_symbol`, `default_symbol_space`, `SymbolAction`, `SymbolDecisionRecord`, `SymbolMessage`, `SymbolReinforcementSignal`, `SymbolAssociation`, `SymbolGroundingLedger`, `SymbolChannel`, `SymbolPolicy`, `__all__` | `DEFER` | Historical opaque-symbol substrate deliberately avoids labels, but that alone does not establish a required TARGET primitive or identity contract. Do not inject external meaning or carry its channel/policy as Organism authority. | Source `test_sequence_substrate.py`, `test_sequence_characterization_ceiling.py`; Lab communication/reproduction and emergent-symbol-grounding consumers. **Closure trigger:** determine whether Biopeia requires a symbolic communication capability independently of the experiment; approve identity, provenance, authority and channel boundaries, then classify; otherwise `DROP`. |
| `telemetry.py`: `MAX_TELEMETRY_EVENTS`, `MAX_EVENT_BYTES`, `MAX_SYMBOLS_PER_EVENT`, `MAX_ID_LENGTH`, `_id`, `_nonnegative`, `CommunicationEvent`, `GroundingEvent`, `CommunicationTelemetry`, `__all__` | `SPLIT` | Source module documents itself as observational and contains no policy/ground truth. Observation/Lab may own event projection, delivery records and experiment telemetry; Organism must not own a telemetry buffer merely because it emitted a communication. Target Observation contract is transport-neutral and must remain passive. | Source `test_communication_telemetry.py`; consumer evidence includes Lab integration/reproduction and communication studies. No direct target counterpart in Organism; classify the evidence stream separately from source state and scientific evidence custody. |
| `temporal.py`: `T`, `TemporalResourceUsage`, `TemporalPrediction`, `TemporalMechanism`, `__all__` | `DEFER` | The source provides a neutral experimental mechanism interface plus resource-use and prediction values; the historical definitions do not prove that this is Organism core. A prediction is not evidence. No exact target temporal mechanism contract exists in this repo. | Source `test_temporal_responsibility.py` and `test_sequence_characterization_ceiling.py`; consumers include Lab `context_tree.py`, `reservoir.py`, temporal evaluation/study. **Closure trigger:** define the required Organism temporal phenomenon/state and separate it from Lab benchmark/resource reporting; then target only the required contract or close as `DROP`. |
| `tokenizer.py`: `_RESERVED`, `_MAX_VOCAB`, `_MAX_SEQUENCE`, `_record_tokens`, `NativeTokenizer` | `SPLIT` | Tokenizer semantics affect model training/inference and artifact reproducibility. Training dataset vocabulary/serialization belongs to Lab's experiment/artifact contract; if Organism inference consumes token IDs, that narrow runtime representation needs a separately versioned target contract. No source tokenizer/API should be imported for compatibility. | Source `test_modeling_tokenizer.py`, `test_private_model_contract.py`; consumers include Lab `dataset.py`, `study.py`, `private_model_training.py` and model architecture code. Target generative adapters do not currently establish equivalence to this tokenizer or its vocabulary/hash. |

### Source blob and last-change provenance

The blob IDs below are resolved from the pinned source tree. “Last change” is
the latest path-changing commit reachable at the pinned source commit, not the
source repository's current branch head.

| Source path | Blob at pinned commit | Last path-changing commit at pin |
|---|---|---|
| `symbiont/src/symbiont/modeling/__init__.py` | `13e676a126bab46ab3285af1f7a0b3b64dd9cfe9` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/authority.py` | `a6aba58f217f284462e188006fff8bb6bb008a3a` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/corpus.py` | `baa311c2c753ae2a94b58f767f09b2d9963fb3d2` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/culture.py` | `a166db9c6a10c4cbec6dff3d5cbaec6db20e7339` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/episodic.py` | `ec54078c071d04e57a13ff35de0aa45a0278d443` | `42ff87fe2102d1a26eb218a5c071ddfb90f42e34` |
| `symbiont/src/symbiont/modeling/experience.py` | `7ea1590ad10385e17f39f446ee1e6e53b933e2cb` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/gateway.py` | `85226509f9a7cff52d3a69bab735c10fb2904507` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/ledger.py` | `02b925fccbdf3c26556bfd48101c3c50eab8494e` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/private_runtime.py` | `dedb99e6106c48c7d49ca6f1101acf2eb69ea80d` | `e2ea9f119f1417171d37f3c7dafcac5653c54a8e` |
| `symbiont/src/symbiont/modeling/proposals.py` | `1d477212c0622afec6bc4a8f308b82c6387e9b7c` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/registry.py` | `2db5fa68afc492a2e63f3b14a93db3a1c916b4e9` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/responsibility.py` | `72a89361f69a4cb113837accd878a1cea7796ab4` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/runtime.py` | `34ea38578282e5ad046d56b9b4692393f61da3b9` | `174983f0c24e1ed9301fe7e1969c8252097cdc54` |
| `symbiont/src/symbiont/modeling/sequences.py` | `fc39aff187074230a602603cf525e01fd36b2b03` | `a05e5aa9a12547a5bc7579845ce2b9f0bcda1f8e` |
| `symbiont/src/symbiont/modeling/symbols.py` | `2fabbc734fdaa0b340d8162a7bad0823f98b2e41` | `42ff87fe2102d1a26eb218a5c071ddfb90f42e34` |
| `symbiont/src/symbiont/modeling/telemetry.py` | `71c88762c2313e28bcfe2a4a0f25427a3e24bbc0` | `fda97e9882100e2d9b0881f63d16f756bc392091` |
| `symbiont/src/symbiont/modeling/temporal.py` | `94ca770a604f39727c8abb5a1b12d8b4261e529b` | `13720935595fedabec2c376dc933629b0deae5aa` |
| `symbiont/src/symbiont/modeling/tokenizer.py` | `cbb235716002589078644c626774a96afbd6d9f7` | `13720935595fedabec2c376dc933629b0deae5aa` |

### Directly relevant source tests and current target evidence

The following pinned source test files were inspected as historical test
coverage. They were not copied wholesale. Test disposition is `DEFER` pending
the matching target contract; after a target slice is justified, rewrite only
the behavior-focused assertions that remain valid, otherwise close the test as
`DROP` with that slice's disposition.

| Pinned source test | Primary source family | Target evidence / limitation |
|---|---|---|
| `symbiont/tests/unit/modeling/test_communication_telemetry.py` | telemetry | No Organism telemetry counterpart; Observation/Lab delivery and evidence contracts remain separate. |
| `symbiont/tests/unit/modeling/test_episodic.py` | episodic | No target `EpisodicExperienceMemory`; target experience/ledger tests cover different bounded records. |
| `symbiont/tests/unit/modeling/test_episodic_projection_trusted.py` | episodic | Trusted projection behavior is not a target contract; do not infer authority from test names. |
| `symbiont/tests/unit/modeling/test_modeling_corpus.py` | corpus | Target memory tests do not exercise scientific dataset selection/split policy; Lab owns any such protocol. |
| `symbiont/tests/unit/modeling/test_modeling_tokenizer.py` | tokenizer | No target tokenizer equivalence or artifact/vocabulary contract established. |
| `symbiont/tests/unit/modeling/test_private_experience_pipeline.py` | experience, ledger, runtime, corpus, gateway | Intentionally not copied; depends on monolithic historical runtime and model-training composition. Target memory and portable-checkpoint tests cover only bounded values and explicit payload composition. |
| `symbiont/tests/unit/modeling/test_private_model_contract.py` | authority, gateway, proposals, registry, culture | Historical contract assertions do not authorize the same policy/architecture in Biopeia. Map each reviewed requirement before rewriting any test. |
| `symbiont/tests/unit/modeling/test_sequence_characterization_ceiling.py` | sequences, symbols, temporal | Not a Biopeia semantic or scientific acceptance threshold. |
| `symbiont/tests/unit/modeling/test_sequence_substrate.py` | sequences, symbols, runtime | No target social-channel/sequence contract; tests stay historical evidence until a target capability is authorized. |
| `symbiont/tests/unit/modeling/test_temporal_responsibility.py` | temporal, responsibility | Target generative tests do not establish equivalent resource metrics, responsibility attribution or scientific comparison. |
| `symbiont/tests/experimental_integrity/test_private_runtime_tick_context.py` | private_runtime, runtime | Historical experimental-integrity gate is not a target runtime gate; rewrite only after Biopeia's causal-time/owner contracts and an integrated runtime exist. |

Source consumer searches at the pinned commit found these direct integration
families: Lab model architecture/artifact/dataset/trainer/study modules under
`lab/src/lab/modeling/`; `lab/src/lab/physics3d/private_model_training.py`;
`lab/src/lab/physics3d/runtime.py`; `lab/src/lab/integration/integrated_habitat.py`;
`lab/src/lab/reproduction/runtime.py`; and Lab learning/communication studies.
These consumers explain historical coupling; they do not establish Biopeia
TARGET ownership. A future code slice must update each actual Biopeia caller,
not keep a compatibility shim for these monorepo imports.

### Scope and closure

This inventory closes the file/symbol classification pass for the pinned
`modeling/` package only. The explicitly provisional `DEFER` rows remain open
work and must be reviewed before migration completion. It does not close the
rest of the historical `symbiont/` source inventory, decide Biopeia's social,
temporal, private-model training, artifact-custody or runtime contracts, prove
that historical source tests pass, or claim a complete Organism continuation.
