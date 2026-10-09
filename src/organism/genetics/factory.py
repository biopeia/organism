"""Factories for canonical Genome v2 starting state."""

from __future__ import annotations

import json
from importlib import resources

from .genome import Genome, GenomeCodec


def create_standard_genome(genome_id: str) -> Genome:
    """Load the repository's canonical Genome v2 seed under a new identity."""
    payload = json.loads(
        resources.files("organism.genetics")
        .joinpath("defaults/base-genome-v2.json")
        .read_text(encoding="utf-8")
    )
    payload["genome_id"] = (
        str(genome_id) if str(genome_id).startswith("genome_") else f"genome_{genome_id}"
    )
    return GenomeCodec().load(payload)
