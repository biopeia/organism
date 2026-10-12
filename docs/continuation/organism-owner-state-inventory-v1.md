# Organism Owner-State Inventory v1

**Evidence base:** `biopeia/organism` commit `833cdb5f14c9cbbef564a54a2f9245278b1bedbd` (2026-10-12)

**Purpose:** record what the current TARGET code demonstrably serializes for the bounded canonical Organism transition/action path, and what remains separate or unproven. This is an implementation inventory for MIG-002; it is not a new architecture decision, a claim of complete Organism continuity, or scientific validation.

## Scope and reading rules

The repository has several independently checkpointable components. A component codec proves only that component's represented state can be encoded/restored under its own contract. It does not prove that a canonical Organism owner checkpoint discovers and composes every state needed by a supported continuation.

The inventory distinguishes:

- **Composed:** included in a named aggregate checkpoint at this evidence revision.
- **External/configuration supplied:** restoration depends on caller-supplied configuration, resolver, schema, or runtime context; this is not necessarily a durable dependency.
- **Separate codec:** the component has its own codec but is not shown composed into the canonical transition checkpoint.
- **Not established:** current evidence does not establish whether the state is required by the supported continuation profile, whether all production callers supply it, or how it is restored atomically.

This document does not infer that every subsystem in the repository must be in every checkpoint. Required state must be determined by the explicitly supported continuation profile and its production callers, not by directory membership.

## Aggregate checkpoints currently present

| Aggregate | Contents demonstrated by source | Explicit boundary / limitation |
|---|---|---|
| `CognitiveRuntimeCheckpoint` schema v9 (`src/organism/cognition/portable_checkpoint.py`) | `CognitiveTransitionState`; cognitive graph structure, learned weights and eligibility via graph codec; supplied `ExperienceRecord` values; optional safety state; sensory normalizers; optional generative resident state; optional all-or-none genome, germline and expression state; effective `KernelLimits` configuration. | Its docstring says it is not a complete cognition or Organism checkpoint. It includes only explicitly supplied state. `dependency_closure` is required to be empty; unsupported external generative dependencies are rejected. |
| `OrganismTransitionState` schema v2 (`src/organism/transition.py`) | `OrganismCheckpoint` identity and opaque private bytes; `organism_step`; optional Embodiment identity reference; optional World causal correlation; explicit lifecycle and lifecycle history. | It is a transition-boundary envelope, not proof of private payload completeness. It delegates dependency verification to the generic envelope's declared dependencies. |
| `OrganismCheckpoint` envelope v1 (`src/organism/checkpoint.py`) | Organism identity, opaque private bytes, content digest and caller-declared dependency references. | The envelope verifies only declared dependencies. It cannot discover omitted owner-private dependencies. |
| Bounded exploration checkpoint v4 (`src/organism/actuation/bounded_action.py`) | Bound transition-state bytes, generation, World source/target ticks, physical step/commit IDs, `CompetenceDevelopmentEngine` checkpoint and `ActionDomain` checkpoint. | This is explicitly a bounded staged exploration/action form. It does not claim complete Organism continuation and is not shown as the general canonical transition aggregate. |
| `OrganismContinuation` schema v1 (`src/organism/continuation.py`) | `continuation_id`, serialized `OrganismTransitionState`, and optional branch ancestry (parent Organism/continuation IDs, parent step and source checkpoint digest). | Portable continuation metadata does not itself confer canonical authority. A branch must use a new Organism identity; ancestry consistency is not source authenticity. |
| `CanonicalContinuationRegistry` schema v2 (`src/organism/continuation.py`) | `authority_id` and canonical records containing Organism identity, continuation ID, step and state digest. | The registry's live lease and identity exclusion guarantees are process-local. Cross-process compare-and-commit must be provided by durable owner storage; copied stores are not authenticated. |

## State inventory

| State family | Current encoding evidence | Restore inputs / constraints | Current composition status and closure gap |
|---|---|---|---|
| Longitudinal identity | `OrganismCheckpoint` envelope; cognitive payload repeats and validates identity; transition envelope validates identity against its owner checkpoint. | Strict identity parsing and equality checks at the shown boundaries. | Included in the respective aggregates; no claim here about canonical-continuation exclusivity across independent registries/processes. |
| Canonical continuation identity and authority | `OrganismContinuation` schema v1 serializes a distinct `continuation_id` and optional ancestry; `CanonicalContinuationRegistry` schema v2 serializes `authority_id` and per-identity canonical step/state digest. | Registry restoration and lease are explicitly process-local. The module requires durable owner storage to perform atomic cross-process compare-and-commit and does not authenticate copied stores. | These are separate continuation/authority codecs, not part of `CognitiveRuntimeCheckpoint` or `OrganismTransitionState`. Local duplicate-lease/stale-state checks do not demonstrate cross-process exclusivity or portable authority. |
| Organism progression and lifecycle | `OrganismTransitionState` stores `organism_step`, explicit lifecycle, lifecycle history, body-context reference and World correlation. | Lifecycle is required; history must be continuous and end at the declared lifecycle. World tick remains correlation, not Organism time. | Included in the transition envelope. Must still be composed atomically with the complete owner-private continuation for MIG-002. |
| Cognitive recurrent state | `CognitiveTransitionState` schema v1: identity, last causal tick and prior activations. | Strict shape, identity, sorted unique activation IDs and finite values. | Composed into cognitive runtime schema v9. |
| Cognitive graph/learned state | `export_graph_checkpoint` / `restore_graph_checkpoint`; tests include restore→continue for the bounded graph/recurrent profile (PR #140). | Graph restoration uses caller-provided limits; effective limits are recorded and must match. | Composed for this profile. This does not establish all learning/model state outside the graph. |
| Experience records | `ExperienceRecord.canonical_payload()` / `restore()` in cognitive runtime payload. | Every record must have matching Organism identity and unique record identity. | Composed only for records supplied to `CognitiveRuntimeCheckpoint`; no evidence here that all production-owned records are passed. |
| Safety and sensory normalization | Optional `safety_state`; `sensory_normalizers` collection in cognitive payload. | Strict component codecs; normalizers are represented regardless of establishment. | Composed when supplied. Production caller completeness remains unproven. |
| Generative resident | Optional resident checkpoint in cognitive payload. | External model registrations or factual-evidence resolver dependencies are unsupported for portable checkpoint; external dependency state is rejected. | Resident-internal durable state is composed only when supplied and self-contained. External models/resolvers are outside this profile. |
| Genetic state | Optional genome, germline and expression state, accepted only as an all-or-none group. | Restore requires `running_version`; genome schema is supplied by caller (default schema otherwise). | Composed when supplied. The running version/schema are restore context, not serialized as a fully self-describing dependency in this payload. Whether genetics is required for the supported continuation profile is not established by this inventory. |
| Effective kernel limits | Cognitive payload stores `graph.kernel_limits` configuration. | Restore requires explicit `KernelLimits` and rejects a mismatch with the recorded configuration. | Composed and validated after PR #141. |
| Agency/action domain | Separate codecs exist for `ActionDomain`, prospective agency, commitments, outcome evidence and related values. | Component-specific restore contracts vary; caller ownership/composition is required. | Not part of `CognitiveRuntimeCheckpoint`. The bounded exploration checkpoint includes an `ActionDomain`, but does not establish all agency/actuation state for general Organism continuation. |
| Competence development and learned motor state | Separate `CompetenceDevelopmentEngine` checkpoint; effective gate and concurrency configuration are represented and validated. | Requires matching Organism/body context; morphology/applicability boundaries are represented by separate contracts. | Included in bounded exploration checkpoint only. General owner-level composition with all required action/competence state is not established. |
| Embodiment/somatic state | Transition envelope holds only an `EmbodimentContextRef` (`body_id`, episode, morphology revision). | Embodiment owns body truth and its state checkpoint. | Somatic state is intentionally not part of Organism private state. A coherent multi-owner cut must preserve owner references outside this Organism aggregate. |
| Transient resolvers, model objects and runtime services | No portable encoding shown for factual resolvers or external model registrations. | Must be rebound/validated by the owning boundary if they become part of a supported profile. | Unsupported by the portable cognitive profile; no silent default/reconstruction claim is made. |

## Restoration guarantees demonstrated at this revision

1. Cognitive runtime decoding rejects unknown/missing top-level fields, duplicate JSON keys, unsupported schema versions, invalid identity joins, invalid graph state, non-empty unsupported dependency closure, and mismatched effective `KernelLimits`.
2. `OrganismCheckpoint` verifies its own private-state digest and each dependency reference it declares. It does not establish that the list is complete.
3. `OrganismTransitionState` requires lifecycle and validates its serialized shape, identity, lifecycle history, progression counter and owner envelope.
4. `OrganismCheckpointContract.restore_transition_checkpoint()` decodes the transition envelope, checks expected Organism identity, and materializes the supported self-contained `CognitiveRuntimeCheckpoint` before returning a `RestoredOrganismTransitionCheckpoint`. The result constructor verifies that its cognitive runtime re-encodes to the exact private checkpoint bytes in the transition envelope. This is a bounded cognition-profile restore, not a general inventory-driven owner composition.
5. The bounded cognition-profile test restores non-default recurrent state, experience records, safety state, sensory normalization and graph state, verifies exact checkpoint re-encoding, then continues through the public transition boundary from the canonical serialized transition. Negative tests cover malformed payload, unsupported dependency closure, wrong identity, size limit, and a result whose materialized runtime does not match its envelope.
6. Bounded exploration restoration checks World committed-cut correlation and restores its engine/action-domain payload. Its source docstring explicitly excludes complete Organism continuation.

## Open acceptance work for MIG-002

This inventory does **not** close `organism#34`, `#41`, or `#45`. The following evidence is still needed:

1. Name the supported continuation profile(s) exercised by production callers, and trace each to the owner boundary that creates and restores it.
2. For each required state in that profile, record whether it is embedded, referenced with a verified immutable identity, or intentionally external configuration. Explain why omitted state cannot alter the promised next transition.
3. Reconcile all separate agency/actuation codecs against the bounded causal-action path; include only state required for that declared profile, but fail closed when any required element is absent.
4. Add an Organism-owned aggregate validation/restore path that validates the entire supported profile before publishing restored state; component-by-component partial restoration must not escape on failure.
5. Test valid restore→next-transition/action continuation and rejection of missing, malformed, inconsistent, unavailable or wrong-identity required state. Preserve exact state where the contract requires it; do not silently initialize defaults.
6. Re-run the inventory against the candidate commit and link each row to source and tests. Keep claims bounded: no World atomicity, cross-process recovery, scientific efficacy, or full runtime completeness absent separate evidence.

## Source index

- `src/organism/cognition/portable_checkpoint.py` — cognitive aggregate schema v9 and strict decoder.
- `src/organism/cognition/transition.py` — recurrent cognitive component state.
- `src/organism/cognition/checkpoint.py` — graph, safety and normalizer codecs.
- `src/organism/checkpoint.py` — opaque owner payload and declared dependency references.
- `src/organism/transition.py` — canonical transition/lifecycle envelope.
- `src/organism/continuation.py` — serialized continuation/ancestry metadata and process-local canonical continuation registry; neither establishes cross-process authority by itself.
- `src/organism/actuation/bounded_action.py` — bounded exploration/action aggregate, owner checkpoint contract, and bounded cognitive-profile restore result; this is not complete owner continuation.
- `src/organism/actuation/domain.py`, `sensorimotor.py`, `competence.py`, `binding.py`, and related modules — independently checkpointable action/competence components; not by themselves the general owner aggregate.
- Tests: `tests/source_unit/cognition/test_portable_checkpoint.py`, `tests/source_unit/cognition/test_owner_transition.py`, `tests/source_unit/actuation/test_exploration_composition.py` (including the bounded cognitive restore profile), `tests/test_continuation.py`, and component checkpoint tests under `tests/source_unit/actuation/`.
