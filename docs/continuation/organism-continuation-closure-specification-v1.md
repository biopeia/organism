# Organism Continuation Closure Specification v1

**Normative reference:** ADR-0081 (*The canonical Organism is longitudinal*), ADR-0103 (*Private learned state belongs to Organism longitudinal continuity*), and `design.cross-domain.longitudinal-continuity-and-reembodiment-v1` (biopeia/docs#147).  
**Tracking issue:** biopeia/organism#161 (relates to open issues #41 and #45; follows up on bounded inventory #34).  
**Authority:** Organism domain owner.  
**Lifecycle:** Proposed specification — precondition for canonical re-embodiment implementation.  

---

## 1. Purpose and Scope

This document specifies the canonical **Organism Continuation Closure** required by Section 5 of the cross-domain Longitudinal Continuity and Re-embodiment Protocol v1.

A canonical re-embodiment preserves the same Organism identity across distinct embodiment episodes and World presences without silent cognitive resets (`CANONICAL_REEMBODIMENT`, ADR-0081). Conversely, an ad-hoc partial state copy constitutes a reduced experimental transplant (`REDUCED_SEED_TRANSPLANT`) and cannot claim canonical continuity.

This specification:
1. Classifies all Organism-owned private states under a strict four-category longitudinal taxonomy.
2. Resolves the architectural boundaries between preserved longitudinal knowledge and somatic executability (#41).
3. Defines fail-closed restoration guarantees for internal and externally resolvable dependencies (#45).
4. Establishes the target composition model for a unified `OrganismLongitudinalCheckpoint`.

---

## 2. Longitudinal State Taxonomy

Every component belonging to `src/organism` is classified under exactly one category:

### A. `REQUIRED_PORTABLE` (Must survive / Internal portable)
* **Definition:** Core subjective state constituting the Organism’s identity, private developmental history, cognitive weights, experiential memory, or genetic baseline.
* **Serialization:** Serialized unconditionally within the Organism owner private payload.
* **Restoration semantics:** Restored bit-exact. Absence, corruption, or digest mismatch causes immediate fail-closed rejection. Silent fresh-state fallback is prohibited.

### B. `SUSPENDED_REVALIDABLE` (Preserved knowledge, suspended execution)
* **Definition:** Learned competencies, action commitments, or motor schemas whose representation was acquired through interaction with a specific physical body morphology and effector set.
* **Serialization:** Serialized bit-exact as longitudinal knowledge.
* **Restoration semantics:** Upon entering a new embodiment episode (or when morphology/actuator context changes), the state is restored into memory, but its executive authorization is marked as `REVALIDATION_REQUIRED` (or suspended). It cannot issue motor commands on new actuators without empirical re-acclimatization.

### C. `RESOLVABLE_EXTERNAL` (External model dependencies)
* **Definition:** High-capacity model weights or factual resolvers that reside outside the in-memory checkpoint payload and are bound via immutable cryptographic content digests (SHA-256).
* **Serialization:** Declared as explicit dependency references in `OrganismCheckpoint.dependencies`.
* **Restoration semantics:** Requires an explicit external resolver (`read_dependency`). Fails closed if the referenced bytes are missing or corrupt. Replacing missing external models with random or default weights is prohibited.

### D. `DISPOSABLE_EPHEMERAL` (Runtime-only machinery)
* **Definition:** Volatile execution artifacts, in-flight IPC message queues, temporary observation buffers, or serialization handles.
* **Serialization:** Excluded by design from durable checkpoints.
* **Restoration semantics:** Re-initialized cleanly upon runtime construction. Omission does not affect longitudinal identity or cognitive continuity.

---

## 3. Comprehensive Component Inventory

| Component / Subsystem | Source Path | Longitudinal Category | Current Codec / Status | Re-embodiment Semantics | Closure Action (#161) |
|---|---|---|---|---|---|
| **Organism Identity** | `src/organism/identity.py` | `REQUIRED_PORTABLE` | Envelope `OrganismCheckpoint` v1 | Immutable across all lifecycles and embodiments. | Enforced by identity equality check. |
| **Progression & Lifecycle** | `src/organism/transition.py` | `REQUIRED_PORTABLE` | Envelope `OrganismTransitionState` v2 | `organism_step` monotonic; records `ACTIVE -> DORMANT -> ACTIVE`. | Composed at transition boundary. |
| **Cognitive Recurrent State** | `src/organism/cognition/transition.py` | `REQUIRED_PORTABLE` | `CognitiveRuntimeCheckpoint` v9 | Prior node activations restored for internal temporal continuity. | Preserved bit-exact. |
| **Cognitive Graph & Weights** | `src/organism/cognition/graph.py` | `REQUIRED_PORTABLE` | Graph codec / `CognitiveRuntimeCheckpoint` v9 | Topology, learned edge weights, and eligibility traces restored bit-exact. | Resolves quantization/rounding audit in #45. |
| **Experiential Memory** | `src/organism/memory/ledger.py`, `experience.py` | `REQUIRED_PORTABLE` | Component codec exists; uncomposed in v9 | Longitudinal memory buffer preserved; records retain epistemic provenance. | Compose `ExperienceLedger` into unified owner aggregate. |
| **Motor Competencies** | `src/organism/actuation/competence.py`, `sensorimotor.py` | `SUSPENDED_REVALIDABLE` | Exploration v4 only; uncomposed in general | Repertoire preserved; execution gated by actuator-context match. | Compose `CompetenceRegistryState` with suspension flag (#41). |
| **Action Domain & Commitments** | `src/organism/actuation/domain.py`, `commitment.py` | `SUSPENDED_REVALIDABLE` | Exploration v4 only | Prior in-flight commitments terminated (`EMBODIMENT_CLOSED`); proposals reset. | Drain active commitments before `DORMANT` transition (#41). |
| **Resident Generative Models** | `src/organism/cognition/generative/resident.py` | `REQUIRED_PORTABLE` (self-contained) | `CognitiveRuntimeCheckpoint` v9 | Internal agenda, hypotheses, and generative budgets restored. | Fails closed if external resolver required without dependency ref. |
| **External Generative Weights** | Generative models with external artifacts | `RESOLVABLE_EXTERNAL` | Rejected in v9 (`unsupported external dependencies`) | Must declare SHA-256 references in `OrganismCheckpoint.dependencies`. | Implement verified resolver path without silent fallbacks (#45). |
| **Genetics & Epigenetics** | `src/organism/genetics/` | `REQUIRED_PORTABLE` | `CognitiveRuntimeCheckpoint` v9 (all-or-none) | Genome, germline, and gene expression state restored together with `running_version`. | Preserved as immutable constitution. |
| **Safety State & Metaplasticity** | `src/organism/cognition/metaplasticity.py` | `REQUIRED_PORTABLE` | `SafetyState` in v9; `MetaParameter` uncomposed | Restored frozen status preserved; parameter limits fail closed if mismatched. | Compose `MetaParameter` into checkpoint (#45). |
| **Embodiment Reference** | `src/organism/transition.py` | `REQUIRED_PORTABLE` (historical record) | Transition envelope v2 | Old context recorded in history; new context adopted on `DORMANT -> ACTIVE`. | Enforced by `accept_organism_transition`. |
| **Runtime Buffers & IO** | Scratch buffers, telemetry queues | `DISPOSABLE_EPHEMERAL` | Unserialized | Purged cleanly upon transition to `DORMANT`. | Excluded by design. |

---

## 4. Key Architectural Invariants

### 4.1 Separation of Knowledge from Executability (#41)
When an Organism transitions from embodiment $E_1$ with morphology $M_1$ to embodiment $E_2$ with morphology $M_2$:
1. Acquired competencies developed under $M_1$ are retained in memory.
2. Any competence bound to actuators absent in $M_2$ or incompatible with $M_2$'s surface fingerprint is marked as `REVALIDATION_REQUIRED`.
3. The Action Domain will reject admission of proposals attempting to execute an unvalidated competence in the new morphology.
4. Competences regain executable status only after empirical sensorimotor calibration in $E_2$.

### 4.2 Unified Longitudinal Checkpoint Target (#34, #41, #45)
To replace the bounded exploration checkpoint v4 and address the gaps in `CognitiveRuntimeCheckpoint` schema v9:
* A target aggregate `OrganismLongitudinalCheckpoint` will compose:
  - `CognitiveRuntimeCheckpoint` (exact graph weights, recurrence, genetics, safety).
  - `ExperienceLedger` (episodic memory).
  - `CompetenceRegistryState` (action repertoire and validation states).
  - Explicit declared external dependency references.
* Restoration of this aggregate must be atomic and fail closed.

---

## 5. Acceptance Criteria for Implementation

1. **Schema Definition:** A versioned aggregate schema capturing the complete `REQUIRED_PORTABLE` and `SUSPENDED_REVALIDABLE` closure.
2. **Fail-Closed Validation:** Strict rejection of absent, corrupt, or inconsistent components across all entry points; removal of permissive `None` fallbacks.
3. **Round-Trip Fidelity:** Exact bitwise equality of cognitive weights, eligibility traces, and memory records across serialization round-trip.
4. **Competence Gating:** Negative tests demonstrating that competencies from morphology $M_1$ cannot be executed in $M_2$ without explicit revalidation.
