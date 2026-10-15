from types import SimpleNamespace

from organism.agency.affordances import AffordanceResolver
from organism.actuation.effects import EffectMatcher, EffectSpace
from organism.actuation.model import EffectPrediction


def test_missing_causal_estimate_does_not_fall_back_to_binding_metric():
    effects = EffectSpace()
    effect = effects.observe({"signal.x": 1.0})
    assert effect is not None

    competence = SimpleNamespace(competence_id="competence.c")
    resolver = AffordanceResolver(
        competences=SimpleNamespace(items=(competence,)),
        predict=lambda **_: EffectPrediction(
            prediction_id="prediction.p",
            competence_id="competence.c",
            context_id=None,
            effect_id=effect.effect_id,
            confidence=0.8,
            support=1,
        ),
        controllability_model=SimpleNamespace(estimate=lambda **_: None),
        execution_bindings=SimpleNamespace(
            get=lambda _: SimpleNamespace(
                surface_fingerprint="surface.s",
                applicability_fingerprint="context.body",
                controllability=0.9,
                reliability=0.7,
                evidence_refs=("evidence.binding",),
            )
        ),
        executable=lambda _: True,
        effect_space=effects,
        effect_matcher=EffectMatcher(),
        surface_fingerprint="surface.s",
        applicability_fingerprint="context.body",
        embodiment_id="body.b",
    )

    affordances = resolver.current(context_ref=None, embodiment_id="body.b")

    assert len(affordances) == 1
    assert affordances[0].controllability == 0.0
    assert affordances[0].executability_confidence == 0.7


def test_affordance_resolver_filters_mismatched_surface_fingerprint():
    effects = EffectSpace()
    effect = effects.observe({"signal.x": 1.0})
    assert effect is not None

    competence = SimpleNamespace(competence_id="competence.c")
    resolver = AffordanceResolver(
        competences=SimpleNamespace(items=(competence,)),
        predict=lambda **_: EffectPrediction(
            prediction_id="prediction.p",
            competence_id="competence.c",
            context_id=None,
            effect_id=effect.effect_id,
            confidence=0.8,
            support=1,
        ),
        controllability_model=SimpleNamespace(estimate=lambda **_: None),
        execution_bindings=SimpleNamespace(
            get=lambda _: SimpleNamespace(
                surface_fingerprint="surface.old",
                controllability=0.9,
                reliability=0.7,
                evidence_refs=("evidence.binding",),
            )
        ),
        executable=lambda _: True,
        effect_space=effects,
        effect_matcher=EffectMatcher(),
        surface_fingerprint="surface.current",
        embodiment_id="body.b",
    )

    affordances = resolver.current(context_ref=None, embodiment_id="body.b")
    assert affordances == ()


def test_affordance_resolver_requires_matching_typed_applicability_context():
    effects = EffectSpace()
    effect = effects.observe({"signal.x": 1.0})
    assert effect is not None
    competence = SimpleNamespace(competence_id="competence.c")
    resolver = AffordanceResolver(
        competences=SimpleNamespace(items=(competence,)),
        predict=lambda **_: EffectPrediction(
            prediction_id="prediction.p",
            competence_id="competence.c",
            context_id=None,
            effect_id=effect.effect_id,
            confidence=0.8,
            support=1,
        ),
        controllability_model=SimpleNamespace(estimate=lambda **_: None),
        execution_bindings=SimpleNamespace(
            get=lambda _: SimpleNamespace(
                surface_fingerprint="surface.same",
                applicability_fingerprint="context.body-a",
                controllability=0.9,
                reliability=0.7,
                evidence_refs=("evidence.binding",),
            )
        ),
        executable=lambda _: True,
        effect_space=effects,
        effect_matcher=EffectMatcher(),
        surface_fingerprint="surface.same",
        applicability_fingerprint="context.body-b",
        embodiment_id="body.b",
    )

    assert resolver.current(context_ref=None, embodiment_id="body.b") == ()


def test_affordance_resolver_fails_closed_when_applicability_context_is_missing():
    effects = EffectSpace()
    effect = effects.observe({"signal.x": 1.0})
    assert effect is not None
    competence = SimpleNamespace(competence_id="competence.c")
    resolver = AffordanceResolver(
        competences=SimpleNamespace(items=(competence,)),
        predict=lambda **_: EffectPrediction(
            prediction_id="prediction.p",
            competence_id="competence.c",
            context_id=None,
            effect_id=effect.effect_id,
            confidence=0.8,
            support=1,
        ),
        controllability_model=SimpleNamespace(estimate=lambda **_: None),
        execution_bindings=SimpleNamespace(
            get=lambda _: SimpleNamespace(
                surface_fingerprint="surface.s",
                applicability_fingerprint=None,
                controllability=0.9,
                reliability=0.7,
                evidence_refs=("evidence.binding",),
            )
        ),
        executable=lambda _: True,
        effect_space=effects,
        effect_matcher=EffectMatcher(),
        surface_fingerprint="surface.s",
        embodiment_id="body.b",
    )

    assert resolver.current(context_ref=None, embodiment_id="body.b") == ()


def test_affordance_resolver_fails_closed_when_binding_has_no_applicability_identity():
    effects = EffectSpace()
    effect = effects.observe({"signal.x": 1.0})
    assert effect is not None
    competence = SimpleNamespace(competence_id="competence.c")
    resolver = AffordanceResolver(
        competences=SimpleNamespace(items=(competence,)),
        predict=lambda **_: EffectPrediction(
            prediction_id="prediction.p",
            competence_id="competence.c",
            context_id=None,
            effect_id=effect.effect_id,
            confidence=0.8,
            support=1,
        ),
        controllability_model=SimpleNamespace(estimate=lambda **_: None),
        execution_bindings=SimpleNamespace(
            get=lambda _: SimpleNamespace(
                surface_fingerprint="surface.s",
                controllability=0.9,
                reliability=0.7,
                evidence_refs=("evidence.binding",),
            )
        ),
        executable=lambda _: True,
        effect_space=effects,
        effect_matcher=EffectMatcher(),
        surface_fingerprint="surface.s",
        applicability_fingerprint="context.body",
        embodiment_id="body.b",
    )

    assert resolver.current(context_ref=None, embodiment_id="body.b") == ()
