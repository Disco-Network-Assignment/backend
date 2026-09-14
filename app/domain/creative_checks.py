"""Creative QA that code can decide: the ad unit's length limits.

Whether the copy makes claims the advertiser never made, or trips the persona's dislikes, is
judgement, so it lives in the copywriter's instructions rather than in a word list. The
copywriter calls these checks through the check_creative tool before finalising, and the
pipeline runs them once more on what came back."""

from app.schemas import CreativeDraft, LintIssue

LIMITS = {"headline": 60, "body": 160, "cta": 20}


def check_lengths(draft: CreativeDraft) -> list[LintIssue]:
    issues = []
    for field_name, limit in LIMITS.items():
        text = getattr(draft, field_name)
        if not text.strip():
            issues.append(LintIssue(rule="length", message=f"{field_name} is empty"))
        elif len(text) > limit:
            issues.append(LintIssue(rule="length", message=f"{field_name} is {len(text)} characters (max {limit})"))
    return issues
