from uuid import UUID

from organism import OrganismIdentity


def test_identity_wraps_explicit_value_without_generating_host_entropy() -> None:
    value = UUID("9fe6ebc6-a82f-4d41-9f65-5e9707a29b05")

    identity = OrganismIdentity.create(value)

    assert identity.value == value
    assert OrganismIdentity.parse(str(identity)) == identity
