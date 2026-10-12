"""Checkpoint serialisation for L8 Prospective Agency.

Persists: OutcomeValueLedger (learned endogenous value statistics), query budget,
and the effective prospective-policy configuration required for continuation.

Not persisted:
- Pending counterfactuals (ephemeral, cross-restart causality breaks)
- Pending deliberation state
- Private frame causal bridge (same reason as _pending_private_frame)

Schema version is checked strictly; unknown versions fail closed.
"""

from __future__ import annotations

PROSPECTIVE_AGENCY_SCHEMA_VERSION = 2

__all__ = [
    "PROSPECTIVE_AGENCY_SCHEMA_VERSION",
]
