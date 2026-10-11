# Cognitive Runtime Checkpoint Decoder Matrix

## Scope and status

This inventory describes the TARGET `CognitiveRuntimeCheckpoint` writer/decoder graph, originally inventoried at implementation base `2ab8775440b94675551bfbd6993afd56df63ba4d` (2026-10-10) and updated for the bounded owner-dependency-closure slice. It is a decoder-field inventory, not a claim that this bounded payload is the complete Organism continuation checkpoint. It does not authorize compatibility with historical Symbiont payloads and does not rename or map any time field.

“Required” means the key is required by the current decoder, including when its value is explicitly `null`. “Explicitly nullable” means the writer/decoder supports a present key whose value is `null`; it never means the key may be omitted. Empty collections are noted separately. Value constraints remain those enforced by each listed decoder and its domain types.

## Root payload: `biopeia.organism.cognitive-runtime`, version 8

All thirteen keys below are required, with no unknown keys accepted.

| Key | Required value / supported absence representation |
|---|---|
| `schema` | Exact string `biopeia.organism.cognitive-runtime`; required, non-null. |
| `schema_version` | Exact integer `8` (booleans rejected); required, non-null. |
| `dependency_closure` | Required empty list. This bounded owner payload supports no external runtime dependencies; registered generative models and factual-evidence resolvers are rejected rather than represented by caller-supplied envelope references. |
| `identity` | Required parseable `OrganismIdentity`; non-null. |
| `transition` | Required `CognitiveTransitionState` payload; non-null. |
| `graph` | Required graph payload; `null` is rejected by the composed decoder. |
| `experience_records` | Required list; empty list is valid. |
| `safety_state` | Required key; explicit `null` is supported and remains `None` after restore; otherwise an exact safety object is restored. |
| `sensory_normalizers` | Required object; empty object is valid. |
| `generative_resident` | Required key; explicit `null` is supported; otherwise resident payload. |
| `genome` | Required key; explicit `null` only when `germline_state` and `expression_state` are also `null`. |
| `germline_state` | Required key; explicit `null` only with the other two genetic keys `null`. |
| `expression_state` | Required key; explicit `null` only with the other two genetic keys `null`. |

The genetic trio is one conditional all-or-none group. If non-null, each must be an object and restore requires an explicit `running_version`. These keys are not individually optional and a partial group is invalid.

## Nested decoder inventory

### Cognitive transition state

`transition` requires exactly `schema`, `schema_version`, `identity`, `last_causal_tick`, and `previous_activations`. The schema/version and identity are required non-null values. `last_causal_tick` is explicitly nullable; otherwise it is a non-negative integer. `previous_activations` is a required list (empty is valid) of exact two-item `[node_id, activation]` entries; IDs are unique and sorted in the restored state.

### Cognitive graph and safety

`graph` requires exactly `weight_codec_version`, `nodes`, and `edges`; the version is required and `nodes`/`edges` are required lists (empty lists are accepted by the codec subject to graph validation).

- Each node requires exactly `node_id`, `kind`, `bias`, `tau`, `predicts_node_id`. `predicts_node_id` is explicitly nullable; the other values are required non-null.
- Each edge requires exactly `source_id`, `target_id`, `kind`, `weight`, `plasticity`, `delay_ticks`, `eligibility`, `support`, `age_ticks`, `stable_ticks`, `last_use_tick`. All are required non-null; numeric/integer/range and graph consistency validation is performed by the graph codec.
- A non-null `safety_state` requires exactly `consecutive_failures` and `frozen`, both non-null; the former is a non-negative integer and the latter a boolean. An explicit `null` is preserved as `None`; it does not construct a `SafetyState()` value.
- Each `sensory_normalizers` map value requires exactly `mean`, `variance`, and `count`, all non-null. The map itself may be empty.

### Experience records

Each `experience_records` list entry requires exactly `record_id`, `organism_id`, `tick_class`, `context_tokens`, `action_token`, `outcome_tokens`, `epistemic_status`, `evidence_refs`, `confidence_class`, and `source_kind`.

- `action_token` is explicitly nullable; its key is still required.
- The three token/reference collections are required lists and may be empty subject to `ExperienceRecord` validation.
- All remaining fields are required non-null and validated by the record type. The composed checkpoint additionally requires each record's organism identity to match the root identity and record IDs to be unique.

### Generative resident

When present, the resident requires exactly `schema_version`, `generative_tick`, `last_symbiont_tick`, `budget`, `agenda`, `scheduler`, `workspace`, `hypotheses`, `calibration`, `generative_use`, `reconciliation_count`, `contradicted_hypothesis_count`, and `factual_contamination_count`.

- `workspace` is explicitly nullable. When non-null, it is a workspace payload.
- `hypotheses` is a required list; empty is valid. Each entry requires `hypothesis`, `target_id`, `predicted_outcomes`, `representation_refs`, and `calibration`. The `calibration` entry is explicitly nullable. The two reference/outcome collections are required lists.
- The remaining resident keys are required non-null and delegated to their component decoder below.
- `last_symbiont_tick` is a required non-negative integer in the current TARGET schema. Its semantic migration/disposition remains unresolved; this matrix does not treat it as `organism_step`, World time, or any other clock.

Generative subcomponents:

| Subcomponent | Required keys | Explicitly nullable / empty |
|---|---|---|
| `budget` | `max_states`, `max_transitions`, `max_depth`, `max_branches`, `max_model_queries` | No nullable fields. |
| `agenda` | `max_targets`, `max_reselection_without_progress`, `suppression_duration`, `agenda_contamination_count`, `targets` | `targets` must be a list; empty is valid. Each target has the exact fields `target_id`, `source`, `source_refs`, `created_tick`, `uncertainty`, `persistence`, `recurrence`, `estimated_resolvability`, `last_selected_tick`, `selection_count`, `last_progress_tick`, `status`, `no_progress_count`, `suppressed_until`; the three `*_tick`/suppression optionals and `estimated_resolvability` accept explicit `null`. |
| `scheduler` | `quotas`, `used` | `used` is a list; empty is valid. Each entry requires `tick`, `mode`, `count`. Quota keys are exactly the current `GenerativeMode` values. |
| `workspace` | wrapper `schema_version`, `workspace`; inner `episode`, `budget`, `states`, `transitions`, `model_queries`, `closed` | Resident-level workspace may be `null`; inside a workspace, collections may be empty. Episode `target_id` and `termination_reason` are explicitly nullable; state `parent_state_id` and generated-feature `value_class`/`source_model_id` are explicitly nullable. No other nullable fields are declared by the current writer/types. |
| `calibration` | `buckets` | List may be empty. Each bucket requires `model_id`, `operation`, `depth_bucket`, `uncertainty_bucket`, `prediction_count`, `comparable_count`, `uncertainty_sum`, `errors`; `errors` is a required list and its length is checked against `comparable_count`. |
| `generative_use` | `schema_version`, `representations`, `eviction_order`, `capacity_pressure`, `sterile_reactivations` | Collections/maps may be empty. Each representation requires `representation_ref`, `activation_count`, `episode_ids`, `source_refs`, `hypothesis_count`, `disagreement_sum`, `disagreement_count`, `last_use_tick`; `last_use_tick` is explicitly nullable. `capacity_pressure` requires `schema_version`, `capacity`, `admissions`, `evictions`, `promotions`, `demotions`, `relearned_after_eviction`, `recently_evicted`. |
| resident hypothesis | `hypothesis_id`, `source_episode_ids`, `source_model_ids`, `uncertainty`, `status`, `factual_support_refs`, `factual_conflict_refs`, `reconciliation_count` | Reference lists may be empty; no field is nullable. |

The non-null `workspace` payload has these exact nested record fields:

- Episode: `episode_id`, `organism_id`, `target_id`, `root_state_id`, `mode`, `started_symbiont_tick`, `started_generative_tick`, `source_episode_ids`, `state_count`, `transition_count`, `branch_count`, `max_depth_reached`, `termination_reason`.
- State: `state_id`, `episode_id`, `origin`, `parent_state_id`, `depth`, `features`, `active_concept_ids`, `relation_refs`, `source_episode_ids`, `source_model_ids`, `source_state_ids`, `uncertainty`, `coherence`, `generative_tick`.
- Feature: `token`, `value_class`, `confidence`, `source_model_id`.
- Transition: `transition_id`, `episode_id`, `source_state_id`, `target_state_id`, `operation`, `model_ids`, `uncertainty_before`, `uncertainty_after`, `predicted_outcomes`, `generative_tick`.
- Workspace budget has the same five required keys listed above.

The resident hypothesis wrapper's non-null calibration object requires exactly `model_id`, `operation`, `depth`, and `uncertainty`.

The registry of generative model owners and the factual evidence resolver are supplied by the restore runtime and are not serialized fields.

### Genetic trio

- Genome: `restore_genome` requires `genome_hash` and `genotype_hash` in addition to the exact GenomeCodec v2 object. GenomeCodec requires `schema_version`, `genome_id`, `kernel_compatibility`, `development`, `plasticity`, `regulation`, `sensorimotor`, `structure`, and `evolvability`. Their exact nested keys are: `development` = `soft_node_budget`, `soft_edge_budget`, `sense_node_budget`, `capacity_growth_sensitivity`, `consolidation_interval_ticks`; `plasticity` = `learning_rate`, `eligibility_decay`, `structural_plasticity`; each adaptive range = `baseline`, `min`, `max`, `adaptation_rate`; `regulation` = `uncertainty_gain`, `novelty_gain`, `prediction_error_gain`, `controllability_loss_gain`, `embodiment_mismatch_gain`, `regulation_smoothing`, `regulation_decay`; `sensorimotor` = `spontaneous_activity_baseline`, `uncertainty_exploration_gain`, `prediction_error_exploration_gain`, `exploration_habituation`, `reacclimation_sensitivity`; `structure` = `growth_threshold`, `pruning_threshold`, `minimum_support`, `tentative_lifetime_ticks`; `evolvability` = `development_mutation_scale`, `plasticity_mutation_scale`, `regulation_mutation_scale`, `sensorimotor_mutation_scale`, `structure_mutation_scale`, `recombination_linkage`. No field may be omitted or `null` under the current schema.
- Germline: exactly `birth_expression`, `inherited_marks`, `acquired_marks`, `acquired_capture_enabled`, and `capture_streaks`. Collections/maps may be empty; no keys are optional. `birth_expression` is validated against the active genome schema's complete flat locus set. Each mark requires `locus`, `delta`, `strength`, `generations_left`.
- Expression: exactly `effective_learning_rate`, `effective_structural_plasticity`, `effective_growth_threshold`, `effective_pruning_threshold`, `exploration_drive`, `birth_expression`, `regulatory_activation`, and `update_count`. `regulatory_activation` is a required object; it must be empty at `update_count == 0` and have the complete current activation key set after updates. It is not nullable.

## Restore inputs not encoded in this payload

The decoder requires explicit `kernel_limits`; genetic restore additionally requires `running_version` when the trio exists and accepts a `genome_schema`; generative restore may receive a model registry. A generic `OrganismCheckpoint` may reference dependencies, in which case restore requires a resolver and validates the declared dependency closure. These are restore prerequisites, not omitted serialized checkpoint fields. Current docs/code explicitly say the model registry and factual resolver are not captured; this matrix does not claim they can be reconstructed from this payload.

## Confirmed gaps and unresolved dispositions

This inspection found no supported evidence for changing the current composed decoder's field requiredness in this slice. Existing writers and strict decoders agree on the exact root and listed nested shapes. Therefore this change records the matrix only; it does not add speculative runtime defaults, schema aliases, or compatibility paths.

The required `last_symbiont_tick` and nested `started_symbiont_tick` fields remain a named unresolved TARGET time-semantics disposition. They are inventoried as current schema requirements, not endorsed as the final canonical Organism clock, and are not mapped here to `organism_step`, World time, or wall-clock time.

The overall payload remains explicitly bounded, not a complete Organism continuation checkpoint. This inventory does not establish that every Organism-owned state component is composed into this checkpoint, that transient external dependencies survive restore, or that issue #60 is complete.

## Source pointers

- `src/organism/cognition/portable_checkpoint.py`
- `src/organism/cognition/transition.py`
- `src/organism/cognition/checkpoint.py`
- `src/organism/memory/experience.py`
- `src/organism/cognition/generative/{resident,agenda,scheduler,workspace,persistence,calibration,consolidation,hypothesis,types}.py`
- `src/organism/genetics/{checkpoint,genome,germline,expression,schema}.py`

## Additional TARGET restore entry points inventoried at the current base

The following production decoder/restore entry points are outside the bounded
`CognitiveRuntimeCheckpoint` graph above. This list records entry points found
in the current TARGET source tree; it does not claim that their complete owner
state is composed into one portable continuation. Field classifications are
recorded here only for the `ActionCommitment` slice covered below. The remaining
entries still require contract-backed field inventories before their restore
coverage can be considered complete.

| Source | Restore entry point | Inventory status |
|---|---|---|
| `src/organism/checkpoint.py` | `OrganismCheckpoint.from_bytes` | Separate envelope/dependency-reference contract; not reclassified here. |
| `src/organism/transition.py` | `OrganismTransitionState.from_bytes` | Separate transition-state envelope; not reclassified here. |
| `src/organism/continuation.py` | `OrganismContinuation.from_bytes`; `CanonicalContinuationRegistry.from_bytes` | Separate continuation/registry envelopes; not reclassified here. |
| `src/organism/capacity.py` | `CapacityPressure.restore` | Component entry point; not reclassified here. |
| `src/organism/provenance.py` | `ProvenanceLog.restore` | Component entry point; not reclassified here. |
| `src/organism/memory/experience.py` | `ExperienceRecord.restore` | Component entry point; not reclassified here. |
| `src/organism/memory/ledger.py` | `ExperienceLedger.restore` | Component entry point; not reclassified here. |
| `src/organism/actuation/commitment.py` | `ActionCommitment.restore` | Bounded required/null/malformed/round-trip test slice in this change. |
| `src/organism/actuation/{binding,causal_evidence,composition,dimension,effects,footprint,intervention,sensorimotor}.py` | Component `restore` methods | Entry points found; per-component field inventory remains outstanding. |
| `src/organism/agency/{executive_outcome,intention,value}.py` | Component `restore` methods | Entry points found; per-component field inventory remains outstanding. |
| `src/organism/agency/prospective.py` | `ProspectiveAgency.restore` | Entry point found; per-component field inventory remains outstanding. |
| `src/organism/cognition/generative/persistence.py` | `restore` | Entry point found; per-component field inventory remains outstanding. |
| `src/organism/genetics/checkpoint.py` | checkpoint/genetic restore functions | Entry points found; matrix above covers the cognitive-runtime genetic trio only. |

### `ActionCommitment.restore` bounded field contract

At the current source base, the writer `ActionCommitment.checkpoint()` emits
exactly the 16 keys listed in `ActionCommitment.restore`; the decoder rejects
missing and unknown keys. The following keys are explicitly nullable in the
writer/type contract and still must be present:

```text
effect_target_id, competence_id, embodiment_id, intent_id,
maximum_duration, ended_tick, end_reason
```

The remaining keys are present and non-null in the current writer contract.
Restore validates schema version, identifier/value types, tick/duration types,
status, and active-versus-terminal fields. Tests exercise omitted required
keys, null for required keys, malformed values, preservation of explicit nulls
for nullable fields, and exact round-trip controls for active and terminal
commitments. This is a test-coverage slice; it does not change the decoder or
assert that every persisted Organism component is covered.

The inventory is intentionally bounded to current TARGET symbols. It does not
import or translate legacy checkpoint fields, and it does not resolve the
separate semantic disposition of historical time names.
