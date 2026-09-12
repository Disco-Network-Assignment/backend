import pytest

from app.domain.lint import CreativeLinter, LintContext
from app.enums import LintSeverity
from tests.helpers import create_sample_creative


@pytest.fixture
def linter():
    return CreativeLinter()


def issues_for(linter, catalog, persona_id="persona_004", description="vet-formulated", others=(), **draft):
    ctx = LintContext(create_sample_creative(**draft), catalog.persona(persona_id), description, others)
    return linter.lint(ctx)


class TestLengthRule:
    def test_clean_draft_passes(self, linter, catalog):
        assert issues_for(linter, catalog) == []

    def test_long_headline_is_hard(self, linter, catalog):
        issues = issues_for(linter, catalog, headline="x" * 61)
        assert [i.rule for i in issues] == ["length"] and issues[0].severity is LintSeverity.HARD

    def test_empty_cta_is_hard(self, linter, catalog):
        assert any(i.message == "cta is empty" for i in issues_for(linter, catalog, cta="  "))


class TestClaimsRule:
    def test_unstated_claim_is_hard(self, linter, catalog):
        issues = issues_for(linter, catalog, body="Clinically proven to cure joint pain.")
        assert {i.message for i in issues} >= {"unsubstantiated claim 'Clinically proven'",
                                                "unsubstantiated claim 'cure'"}

    def test_claim_stated_by_advertiser_is_allowed(self, linter, catalog):
        issues = issues_for(linter, catalog, description="Our food is clinically proven.",
                            body="Clinically proven senior nutrition.")
        assert issues == []


class TestPersonaDisinterestRule:
    def test_trendy_language_trips_affluent_classic(self, linter, catalog):
        issues = issues_for(linter, catalog, persona_id="persona_005", headline="The viral bag everyone is obsessed with")
        assert any(i.rule == "persona_disinterest" and "trendy language" in i.message for i in issues)

    def test_vague_eco_claim_trips_sustainability_buyer_unless_specific(self, linter, catalog):
        vague = issues_for(linter, catalog, persona_id="persona_006", body="An eco-friendly choice.")
        assert any("vague eco claims" in i.message for i in vague)
        specific = issues_for(linter, catalog, persona_id="persona_006",
                              body="An eco-friendly choice: 100% recycled bottles.")
        assert not any("vague eco claims" in i.message for i in specific)

    def test_subscription_wording_trips_gifter_unless_gift_framed(self, linter, catalog):
        plain = issues_for(linter, catalog, persona_id="persona_010", cta="Subscribe now")
        assert any("subscription-only" in i.message for i in plain)
        framed = issues_for(linter, catalog, persona_id="persona_010", cta="Subscribe now",
                            body="Give the first box as a gift.")
        assert not any("subscription-only" in i.message for i in framed)


class TestSoftRules:
    def test_duplicate_headline_is_soft(self, linter, catalog):
        issues = issues_for(linter, catalog, others=("joint support they will actually eat",))
        assert issues and all(i.severity is LintSeverity.SOFT for i in issues)
        assert CreativeLinter.passed(issues)

    def test_shouting_is_soft(self, linter, catalog):
        issues = issues_for(linter, catalog, headline="HUGE DEAL TODAY!! Really!")
        assert {i.message for i in issues} == {"more than one ALL-CAPS word", "more than one exclamation mark"}
