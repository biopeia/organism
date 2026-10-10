from dataclasses import replace
from uuid import UUID

import pytest

from organism.cognition import (
    CognitiveTransitionError,
    CognitiveTransitionState,
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    advance_cognition_from_sense_signals,
)
from organism.cognition.sense_input import (
    SCALAR_FIELD_CONTRACT_ID,
    SCALAR_FIELD_CONTRACT_REVISION,
    SCALAR_FIELD_QUANTITY_KIND,
    SCALAR_FIELD_UNIT,
)
from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity


def _graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=(
            PlasticNode("sense.a", NodeKind.SENSE),
            PlasticNode("sense.b", NodeKind.SENSE),
            PlasticNode("state", NodeKind.STATE),
        ),
        edges=(
            PlasticEdge("sense.a", "state", EdgeKind.EXCITATORY, 1.0, 0.0, 0),
            PlasticEdge("sense.b", "state", EdgeKind.EXCITATORY, 1.0, 0.0, 0),
        ),
        kernel_limits=KernelLimits(),
    )


def _provenance(**overrides: object) -> SenseProvenance:
    values: dict[str, object] = {
        "world_id": "world-1",
        "causal_tick": 12,
        "physical_step_id": "physical-step-12",
        "environment_id": "environment-1",
        "environment_revision": "environment-rev-4",
        "sample_frame_id": "world-frame-1",
        "sample_position_m": (1.25, -0.5, 0.75),
        "body_id": "body-1",
        "embodiment_episode_id": "episode-2",
        "morphology_revision": "morph-7",
    }
    values.update(overrides)
    return SenseProvenance(**values)  # type: ignore[arg-type]


def _state() -> CognitiveTransitionState:
    return CognitiveTransitionState.initial(
        OrganismIdentity.create(UUID("00000000-0000-0000-0000-000000000123"))
    )


def _signal(channel_id: str, value: float, provenance: SenseProvenance) -> OpaqueSenseSignal:
    return OpaqueSenseSignal(
        channel_id=channel_id,
        receptor_id={
            "channel-left": "receptor-left",
            "channel-right": "receptor-right",
            "other-channel": "other-receptor",
        }[channel_id],
        value=value,
        provenance=provenance,
        contract_id=SCALAR_FIELD_CONTRACT_ID,
        contract_revision=SCALAR_FIELD_CONTRACT_REVISION,
        quantity_kind=SCALAR_FIELD_QUANTITY_KIND,
        unit=SCALAR_FIELD_UNIT,
    )


def _bindings() -> tuple[SenseChannelBinding, ...]:
    return (
        SenseChannelBinding("channel-left", "receptor-left", "sense.a"),
        SenseChannelBinding("channel-right", "receptor-right", "sense.b"),
    )


def test_opaque_signals_bind_by_stable_channel_identity_not_collection_order() -> None:
    graph = _graph()
    provenance = _provenance()
    left = _signal("channel-left", 0.2, provenance)
    right = _signal("channel-right", 0.7, provenance)

    state_a, frame_a = advance_cognition_from_sense_signals(
        _state(), graph, _bindings(), (left, right), expected_provenance=provenance
    )
    state_b, frame_b = advance_cognition_from_sense_signals(
        _state(), graph, tuple(reversed(_bindings())), (right, left),
        expected_provenance=provenance,
    )

    assert frame_a.activations == frame_b.activations
    assert frame_a.activations["sense.a"] == 0.2
    assert frame_a.activations["sense.b"] == 0.7
    assert state_a == state_b


def test_missing_signal_is_not_silently_converted_to_zero() -> None:
    provenance = _provenance()
    with pytest.raises(CognitiveTransitionError, match="missing configured"):
        advance_cognition_from_sense_signals(
            _state(), _graph(), _bindings(),
            (_signal("channel-left", 0.2, provenance),),
            expected_provenance=provenance,
        )


def test_unconfigured_or_duplicate_channels_are_rejected() -> None:
    provenance = _provenance()
    with pytest.raises(CognitiveTransitionError, match="unconfigured"):
        advance_cognition_from_sense_signals(
            _state(), _graph(), _bindings(),
            (_signal("other-channel", 0.2, provenance), _signal("channel-right", 0.7, provenance)),
            expected_provenance=provenance,
        )
    with pytest.raises(CognitiveTransitionError, match="duplicate signal"):
        advance_cognition_from_sense_signals(
            _state(), _graph(), _bindings(),
            (_signal("channel-left", 0.2, provenance), _signal("channel-left", 0.3, provenance)),
            expected_provenance=provenance,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_id", "other-world"),
        ("causal_tick", 11),
        ("physical_step_id", "other-step"),
        ("environment_id", "other-environment"),
        ("environment_revision", "stale-revision"),
        ("sample_frame_id", "other-frame"),
        ("sample_position_m", (1.25, -0.5, 0.76)),
        ("body_id", "other-body"),
        ("embodiment_episode_id", "old-episode"),
        ("morphology_revision", "old-morphology"),
    ],
)
def test_stale_or_mismatched_source_provenance_is_rejected(field: str, value: object) -> None:
    expected = _provenance()
    signal = _signal("channel-left", 0.2, replace(expected, **{field: value}))
    with pytest.raises(CognitiveTransitionError, match="provenance"):
        advance_cognition_from_sense_signals(
            _state(), _graph(), _bindings(),
            (signal, _signal("channel-right", 0.7, expected)),
            expected_provenance=expected,
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.4", 10**400])
def test_invalid_scalar_values_fail_closed(value: object) -> None:
    with pytest.raises(CognitiveTransitionError):
        _signal("channel-left", value, _provenance())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sample_position_m",
    [
        (0.0, 1.0),
        (0.0, 1.0, float("inf")),
        (0.0, 1.0, True),
        (0.0, 1.0, 10**400),
    ],
)
def test_invalid_sample_position_fails_closed(
    sample_position_m: tuple[object, ...],
) -> None:
    with pytest.raises(CognitiveTransitionError, match="sample_position_m"):
        _provenance(sample_position_m=sample_position_m)


def test_unbound_sense_node_and_non_sense_binding_target_are_rejected() -> None:
    provenance = _provenance()
    only_left = (
        SenseChannelBinding("channel-left", "receptor-left", "sense.a"),
    )
    with pytest.raises(CognitiveTransitionError, match="cover exactly"):
        advance_cognition_from_sense_signals(
            _state(), _graph(), only_left,
            (_signal("channel-left", 0.2, provenance),),
            expected_provenance=provenance,
        )


def test_channel_binding_rejects_mismatched_stable_receptor_identity() -> None:
    provenance = _provenance()
    signal = OpaqueSenseSignal(
        channel_id="channel-left",
        receptor_id="other-receptor",
        value=0.2,
        provenance=provenance,
        contract_id=SCALAR_FIELD_CONTRACT_ID,
        contract_revision=SCALAR_FIELD_CONTRACT_REVISION,
        quantity_kind=SCALAR_FIELD_QUANTITY_KIND,
        unit=SCALAR_FIELD_UNIT,
    )
    with pytest.raises(CognitiveTransitionError, match="mismatched receptor"):
        advance_cognition_from_sense_signals(
            _state(), _graph(), _bindings(),
            (signal, _signal("channel-right", 0.7, provenance)),
            expected_provenance=provenance,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract_id", "unrecognized-transduction"),
        ("contract_revision", 2),
        ("contract_revision", True),
        ("quantity_kind", "temperature"),
        ("unit", "m/s"),
    ],
)
def test_incompatible_semantic_contract_is_rejected(field: str, value: object) -> None:
    provenance = _provenance()
    signal = _signal("channel-left", 0.2, provenance)
    with pytest.raises(CognitiveTransitionError):
        replace(signal, **{field: value})

    wrong = (
        SenseChannelBinding("channel-left", "receptor-left", "state"),
        SenseChannelBinding("channel-right", "receptor-right", "sense.b"),
    )
    with pytest.raises(CognitiveTransitionError, match="declared SENSE"):
        advance_cognition_from_sense_signals(
            _state(), _graph(), wrong,
            (_signal("channel-left", 0.2, provenance), _signal("channel-right", 0.7, provenance)),
            expected_provenance=provenance,
        )
