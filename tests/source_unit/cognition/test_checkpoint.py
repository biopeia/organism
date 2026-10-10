from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from organism.cognition.checkpoint import export_genome_checkpoint, restore_genome_checkpoint
from organism.genetics.checkpoint import restore_genome as restore_genome_genetics
from organism.cognition.genome import GenomeCodec, GenomeError
from organism.cognition.limits import KernelLimits

from .test_genome import VALID_PAYLOAD

_RUNNING_VERSION = (0, 85, 0)


def test_none_genome_round_trips_to_none():
    assert export_genome_checkpoint(None) is None
    assert (
        restore_genome_checkpoint(
            None, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
        )
        is None
    )


def test_valid_genome_round_trips_exactly():
    genome = GenomeCodec().load(VALID_PAYLOAD)
    payload = export_genome_checkpoint(genome)
    restored = restore_genome_checkpoint(
        payload, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
    )
    assert restored == genome
    assert restored.genome_hash == genome.genome_hash


def test_exported_payload_carries_both_required_genome_hashes():
    genome = GenomeCodec().load(VALID_PAYLOAD)
    payload = export_genome_checkpoint(genome)
    assert payload["genome_hash"] == genome.genome_hash
    assert payload["genotype_hash"] == genome.genotype_hash


@pytest.mark.parametrize("invalid_value", [None, 1, ""])
def test_cognition_and_genetics_codecs_both_reject_invalid_genotype_hash(invalid_value):
    genome = GenomeCodec().load(VALID_PAYLOAD)
    payload = export_genome_checkpoint(genome)
    payload["genotype_hash"] = invalid_value
    with pytest.raises(GenomeError):
        restore_genome_checkpoint(
            payload, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
        )
    with pytest.raises(GenomeError):
        restore_genome_genetics(
            payload, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
        )


def test_cognition_and_genetics_codecs_both_reject_missing_genotype_hash():
    genome = GenomeCodec().load(VALID_PAYLOAD)
    payload = export_genome_checkpoint(genome)
    del payload["genotype_hash"]
    with pytest.raises(GenomeError):
        restore_genome_checkpoint(
            payload, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
        )
    with pytest.raises(GenomeError):
        restore_genome_genetics(
            payload, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
        )


def test_tampered_payload_is_rejected_on_restore():
    genome = GenomeCodec().load(VALID_PAYLOAD)
    payload = export_genome_checkpoint(genome)
    tampered = copy.deepcopy(payload)
    tampered["plasticity"]["eligibility_decay"] = 0.01
    with pytest.raises(GenomeError):
        restore_genome_checkpoint(
            tampered, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
        )


def test_restore_rejects_a_genome_that_no_longer_satisfies_kernel_limits():
    genome = GenomeCodec().load(VALID_PAYLOAD)
    payload = export_genome_checkpoint(genome)
    with pytest.raises(GenomeError):
        restore_genome_checkpoint(
            payload, kernel_limits=KernelLimits(max_nodes=1), running_version=_RUNNING_VERSION
        )


def test_restore_does_not_use_legacy_kernel_validation_for_old_genomes():
    genome = GenomeCodec().load(VALID_PAYLOAD)
    historical_genome = replace(genome, kernel_compatibility=">=0.55,<0.60")
    payload = export_genome_checkpoint(historical_genome)

    with pytest.raises(GenomeError, match="does not admit running version"):
        restore_genome_checkpoint(
            payload, kernel_limits=KernelLimits(), running_version=_RUNNING_VERSION
        )
