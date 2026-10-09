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
- an opaque, in-memory checkpoint boundary with verified external dependency
  references;
- copied cognition, learning, agency, genetics and actuation capabilities are
  still subject to migration review; their presence does not validate their
  scientific efficacy;
- no World recovery protocol or cross-authority persistence format is selected.

## Development

```sh
python -m pip install -e '.[dev]'
pytest
```

Package import name: `organism`.
