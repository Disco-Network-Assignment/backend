"""Creative QA. The copywriter is asked to respect lengths, avoid claims the advertiser never
made, and avoid what the persona is disinterested in; this checks that it did. Hard issues
trigger one regeneration with the issues quoted back; soft issues become warnings."""

import re
from dataclasses import dataclass
from typing import Protocol

from app.enums import LintSeverity
from app.schemas import CreativeDraft, LintIssue, ShopperPersona

HEADLINE_MAX = 60
BODY_MAX = 160
CTA_MAX = 20


@dataclass(frozen=True)
class LintContext:
    draft: CreativeDraft
    persona: ShopperPersona
    description: str  # the advertiser text: claims it states are allowed


class LintRule(Protocol):
    name: str

    def check(self, ctx: LintContext) -> list[LintIssue]: ...


class LengthRule:
    name = "length"

    def check(self, ctx: LintContext) -> list[LintIssue]:
        issues = []
        for field_name, text, limit in (("headline", ctx.draft.headline, HEADLINE_MAX),
                                        ("body", ctx.draft.body, BODY_MAX),
                                        ("cta", ctx.draft.cta, CTA_MAX)):
            if not text.strip():
                issues.append(hard(self.name, f"{field_name} is empty"))
            elif len(text) > limit:
                issues.append(hard(self.name, f"{field_name} is {len(text)} characters (max {limit})"))
        return issues


class ClaimsRule:
    """Claims that need substantiation are only allowed when the advertiser stated them."""

    name = "claims"
    pattern = re.compile(
        r"\b(cures?|cured|guaranteed|clinically proven|doctor[- ]recommended|#1|number one|"
        r"miracle|100%|scientifically proven|fda[- ]approved)\b", re.I)

    def check(self, ctx: LintContext) -> list[LintIssue]:
        text = f"{ctx.draft.headline} {ctx.draft.body} {ctx.draft.cta} {ctx.draft.alt_headline}"
        stated = ctx.description.lower()
        return [hard(self.name, f"unsubstantiated claim '{m.group(0)}'")
                for m in self.pattern.finditer(text) if m.group(0).lower() not in stated]


class PersonaDisinterestRule:
    """Maps the data pack's `disinterested_in` phrases to the words that betray them. Phrases
    without a pattern are matters of judgement left to the model and the reviewer."""

    name = "persona_disinterest"
    patterns: dict[str, re.Pattern[str]] = {
        "trendy language": re.compile(r"\b(trendy|viral|slay|obsessed|iconic|vibes?)\b", re.I),
        "loud aesthetics": re.compile(r"\b(bold|loud|neon|statement)\b", re.I),
        "influencer positioning": re.compile(r"\b(influencer|creator|as seen on|tiktok)\b", re.I),
        "luxury positioning": re.compile(r"\b(luxury|luxe|exclusive|elite|indulgent)\b", re.I),
        "luxury-only messaging": re.compile(r"\b(luxury|luxe|exclusive|elite)\b", re.I),
        "vague premium positioning": re.compile(r"\b(premium quality|world-class|finest)\b", re.I),
        "ultra-cheap positioning": re.compile(r"\b(cheapest|dirt cheap|bargain bin|rock bottom)\b", re.I),
        "generic pet brands": re.compile(r"\b(generic|no-name|store brand)\b", re.I),
        "corporate voice": re.compile(r"\b(synerg\w*|leverag\w*|solutions?|stakeholders?)\b", re.I),
        "novelty": re.compile(r"\b(revolutionary|game-changing|all-new|breakthrough)\b", re.I),
        "vague eco claims": re.compile(r"\b(eco-friendly|green|earth-friendly|planet-friendly)\b", re.I),
        "subscription-only": re.compile(r"\b(subscribe|subscription|monthly plan)\b", re.I),
        "excessive packaging": re.compile(r"\b(gift box(?:ed)?|lavish(?:ly)? wrapped)\b", re.I),
    }
    # a vague eco word is fine next to a specific claim; a subscription is fine framed as a gift
    specific_eco = re.compile(r"\b(recycled|refill\w*|certified|carbon|plastic|%|compostable|b corp|fsc|organic)\b", re.I)
    gift_framing = re.compile(r"\b(gift|gifting|give|treat someone)\b", re.I)

    def check(self, ctx: LintContext) -> list[LintIssue]:
        text = f"{ctx.draft.headline} {ctx.draft.body} {ctx.draft.cta}"
        issues = []
        for phrase in ctx.persona.disinterested_in:
            pattern = self.patterns.get(phrase.lower())
            match = pattern.search(text) if pattern else None
            if match is None:
                continue
            if phrase == "vague eco claims" and self.specific_eco.search(text):
                continue
            if phrase == "subscription-only" and self.gift_framing.search(text):
                continue
            issues.append(hard(self.name, f"'{match.group(0)}' trips {ctx.persona.name}'s disinterest in {phrase}"))
        return issues


class StyleRule:
    name = "style"
    caps_word = re.compile(r"\b[A-Z]{4,}\b")

    def check(self, ctx: LintContext) -> list[LintIssue]:
        text = f"{ctx.draft.headline} {ctx.draft.body}"
        issues = []
        if len(self.caps_word.findall(text)) > 1:
            issues.append(soft(self.name, "more than one ALL-CAPS word"))
        if text.count("!") > 1:
            issues.append(soft(self.name, "more than one exclamation mark"))
        return issues


DEFAULT_RULES: tuple[LintRule, ...] = (LengthRule(), ClaimsRule(), PersonaDisinterestRule(), StyleRule())


class CreativeLinter:
    def __init__(self, rules: tuple[LintRule, ...] = DEFAULT_RULES) -> None:
        self._rules = rules

    def lint(self, ctx: LintContext) -> list[LintIssue]:
        return [issue for rule in self._rules for issue in rule.check(ctx)]

    @staticmethod
    def passed(issues: list[LintIssue]) -> bool:
        return not any(issue.severity is LintSeverity.HARD for issue in issues)


def hard(rule: str, message: str) -> LintIssue:
    return LintIssue(severity=LintSeverity.HARD, rule=rule, message=message)


def soft(rule: str, message: str) -> LintIssue:
    return LintIssue(severity=LintSeverity.SOFT, rule=rule, message=message)
