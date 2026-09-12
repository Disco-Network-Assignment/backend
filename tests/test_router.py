from app.domain.router import InputRouter
from app.enums import InputQuality, RouteFlag
from tests.helpers import create_sample_brief


class TestInputRouter:
    def test_junk_is_trivially_insufficient(self):
        router = InputRouter()
        assert router.is_trivially_insufficient("idk")
        assert router.is_trivially_insufficient("test")
        assert router.is_trivially_insufficient("   ")
        assert not router.is_trivially_insufficient("We sell premium dog food")

    def test_insufficient_stops(self):
        decision = InputRouter().route(create_sample_brief(input_quality=InputQuality.INSUFFICIENT))
        assert decision.stop and decision.reason

    def test_clear_runs_untouched(self):
        decision = InputRouter().route(create_sample_brief())
        assert not decision.stop and decision.flags == () and decision.persona_cap == 5
        assert decision.confidence_multiplier == 1.0

    def test_vague_continues_with_caveats(self):
        decision = InputRouter().route(create_sample_brief(input_quality=InputQuality.VAGUE))
        assert decision.flags == (RouteFlag.ASSUMPTIONS,)
        assert decision.persona_cap == 3 and decision.confidence_multiplier == 0.7

    def test_ambiguous_surfaces_interpretations(self):
        decision = InputRouter().route(create_sample_brief(input_quality=InputQuality.AMBIGUOUS))
        assert RouteFlag.INTERPRETATIONS in decision.flags

    def test_non_consumer_is_off_catalog_even_if_labelled_clear(self):
        decision = InputRouter().route(create_sample_brief(is_consumer_commerce=False))
        assert RouteFlag.OFF_CATALOG in decision.flags and decision.confidence_multiplier == 0.5
