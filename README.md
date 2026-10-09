# Biopeia Organism

The Organism package owns longitudinal identity and private Organism state. It
does not own physical body state, environmental truth, or World causal
coordination.

This repository begins the incremental extraction from the historical
`symbiont-lab` repository. The source remains intact. See
[`MIGRATION_MAP.md`](MIGRATION_MAP.md) for the first bounded source mapping.

## Scope of this initial slice

- immutable Organism identity values, including explicit fresh identities for
  branches;
- an opaque, in-memory checkpoint boundary with verified external dependency
  references;
- no runtime, cognition implementation, file format, or World recovery
  protocol is selected here.

## Development

```sh
python -m pip install -e '.[dev]'
pytest
```

Package import name: `biopeia_organism`.
