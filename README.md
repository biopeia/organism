# Biopeia Organism

The Organism package owns longitudinal identity and private Organism state. It
does not own physical body state, environmental truth, or World causal
coordination.

This repository is an incremental extraction from the historical
`symbiont-lab` repository. The source remains intact. See
[`MIGRATION.md`](MIGRATION.md) for source mapping, exclusions and provenance.

## Scope of this initial slice

- immutable Organism identity values, including explicit fresh identities for
  branches;
- an explicit continuation envelope carrying a stable `continuation_id`,
  Organism step and optional branch ancestry; the envelope itself never claims
  canonical authority;
- an Organism-owned compare-and-commit registry API that rejects a second live
  restore and stale canonical checkpoints within one process-local registry;
  the registry snapshot is owner state, not a portable authority token, and
  this implementation does not provide cross-process coordination or global
  uniqueness across copied stores;
- an explicit, persisted Organism lifecycle (`ACTIVE`, `DORMANT`, `TERMINATED`)
  with transition provenance; lifecycle changes do not advance `organism_step`,
  and missing/invalid lifecycle data fails closed. A `DORMANT → ACTIVE`
  reactivation may bind a new identity-only Embodiment context without
  advancing the Organism step. Provenance references are caller-supplied and
  do not establish authentication, authorization, or trigger validity;
- a branch operation that requires owner-produced private state under a new
  `organism_id`, records source checkpoint provenance and requires explicit
  registry designation before the fork becomes canonical;
- an owner-opaque, schema-versioned checkpoint byte envelope with integrity
  checks and verified external dependency references;
- a bounded cognition component checkpoint for recurrent activations and their
  explicit causal tick, nestable inside the opaque Organism payload;
- copied cognition, learning, agency, genetics and actuation capabilities are
  still subject to migration review; their presence does not validate their
  scientific efficacy;
- the envelope does not assemble the full runtime state and does not select a
  World recovery protocol or cross-authority persistence format.
- `CognitiveRuntimeCheckpoint` composes graph structure/quantized learned
  weights with recurrent activations inside the generic Organism envelope.
  It excludes labile eligibility, private model artifacts and other cognition
  components; it is not a complete Organism checkpoint.
- The first scalar-field SENSE boundary accepts only explicitly bound opaque
  channel/receptor identities, dimensionless `[0, 1]` values, and matching
  causal/source/body applicability, including the declared sampling frame and
  meter-based position. Missing configured signals are rejected; the boundary
  does not assign environmental meaning or prove learning.

## Development

```sh
python -m pip install -e '.[dev]'
pytest
```

Package import name: `organism`.
