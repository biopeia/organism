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
- an owner-opaque, schema-versioned checkpoint byte envelope with integrity
  checks and verified external dependency references;
- a bounded cognition component checkpoint for recurrent activations and their
  explicit causal tick, nestable inside the opaque Organism payload;
- copied cognition, learning, agency, genetics and actuation capabilities are
  still subject to migration review; their presence does not validate their
  scientific efficacy;
- the envelope does not assemble the full runtime state and does not select a
  World recovery protocol or cross-authority persistence format.
- the cognition component checkpoint is not a complete cognition or Organism
  checkpoint; graph/learned-model and other owner state remain to be composed.

## Development

```sh
python -m pip install -e '.[dev]'
pytest
```

Package import name: `organism`.
