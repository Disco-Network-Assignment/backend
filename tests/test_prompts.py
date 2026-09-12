import pytest

from app.prompts.registry import PromptError, PromptRegistry

# every prompt with the variables the executor renders it with (mirrors prompts/README.md)
DECLARED = {
    "intake": {"categories": "a, b", "attributes": "x, y", "description": "We sell dog food."},
    "match_publishers": {"catalog": [{"id": "pub_001"}], "brief": {"a": 1}, "signals": [], "publisher_count": "20"},
    "select_personas": {"personas": [], "persona_cap": "5", "brief": {}, "recommended": []},
    "write_creative": {"brief": {}, "persona": {}, "angle": "a", "watchouts": "w", "target_publishers": "p", "feedback": ""},
    "campaign_summary": {"plan": {"budget": 1}},
    "validation_retry": {"errors": "- missing"},
    "lint_retry": {"issues": "- too long"},
}


class TestPromptRegistry:
    def test_loads_every_prompt(self, prompts):
        assert set(prompts.names) == set(DECLARED)

    @pytest.mark.parametrize("name", sorted(DECLARED))
    def test_renders_with_declared_variables(self, prompts, name):
        rendered = prompts.render(name, **DECLARED[name])
        assert "{{" not in rendered.input and "{{" not in (rendered.instructions or "")
        assert rendered.version

    def test_stage_prompts_have_instructions(self, prompts):
        for name in ("intake", "match_publishers", "select_personas", "write_creative", "campaign_summary"):
            assert prompts.get(name).system

    def test_objects_render_as_json(self, prompts):
        rendered = prompts.render("campaign_summary", plan={"budget": {"total": 5000}})
        assert '"total": 5000' in rendered.input

    def test_missing_variable_raises(self, prompts):
        with pytest.raises(PromptError, match="missing"):
            prompts.render("intake", categories="a", attributes="b")

    def test_unknown_variable_raises(self, prompts):
        with pytest.raises(PromptError, match="unknown"):
            prompts.render("campaign_summary", plan={}, extra=1)

    def test_unknown_prompt_raises(self, prompts):
        with pytest.raises(PromptError):
            prompts.get("nope")

    def test_advertiser_text_is_wrapped_as_data(self, prompts):
        rendered = prompts.render("intake", **DECLARED["intake"])
        assert "<advertiser_description>\nWe sell dog food.\n</advertiser_description>" in rendered.input

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(PromptError):
            PromptRegistry(tmp_path)
