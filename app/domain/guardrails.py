"""Rules a model verdict must never break.

The matcher agent scores every publisher and gives a verdict. These rules run afterwards:

    - every publisher in the catalog is accounted for exactly once
    - scores stay inside 0-100
    - an off-catalog advertiser never gets a "recommend", and its scores are capped
    - a product priced far above what a publisher's shoppers spend is capped
    - a "recommend" needs a score to match; the recommended set is bounded (3 to 6)
    - a verdict that disagrees wildly with the computed prior is flagged

Every rule that fires leaves a GuardrailTag on the assessment so the UI can show it.
"""

from dataclasses import dataclass, field

from app.domain.catalog import CatalogRepository
from app.enums import GuardrailTag, IncomeTier, PurchaseModel, Verdict
from app.schemas import (
    AdvertiserBrief,
    FitSignals,
    MatchOutput,
    PublisherAssessment,
    PublisherAssessmentDraft,
    Subscores,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardConfig:
    recommend_min_score: float = 70.0   # a "recommend" needs at least this to stay in the top set
    promote_min_score: float = 60.0     # a "consider" above this can be promoted to fill the minimum
    max_recommended: int = 6            # more than 6 is not a recommendation, it is a list
    min_recommended: int = 3            # fewer than 3 gives the advertiser nothing to compare
    off_catalog_cap: float = 40.0       # off-catalog advertisers never look like a good fit
    aov_mismatch_ratio: float = 6.0     # price more than 6x the publisher's AOV...
    aov_mismatch_cap: float = 55.0      # ...caps the score, except on high-income publishers
    divergence_from_prior: float = 40.0 # a score this far from the computed prior gets flagged
    max_missing_before_retry: int = 3   # more omissions than this -> ask the model again


# sort order for ranking: recommend first, then consider, then exclude
VERDICT_ORDER = {Verdict.RECOMMEND: 0, Verdict.CONSIDER: 1, Verdict.EXCLUDE: 2}

NOT_ASSESSED_REASON = "Not assessed by the model; excluded by default."
BEYOND_TOP_SET_REASON = "Not in the recommended set after ranking."


@dataclass
class _Row:
    """One publisher while the rules are being applied; becomes a PublisherAssessment."""

    draft: PublisherAssessmentDraft
    signals: FitSignals
    score: float
    verdict: Verdict
    tags: list[GuardrailTag] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class AssessmentGuard:
    def __init__(self, catalog: CatalogRepository, config: GuardConfig = GuardConfig()):
        self.catalog = catalog
        self.config = config

    # ---- validation: decides whether the runner asks the model again ----

    def validation_errors(self, output: MatchOutput) -> list[str]:
        """Problems worth a retry: too many publishers skipped, duplicates, missing reasons."""
        errors = []
        seen_ids = []
        for assessment in output.assessments:
            seen_ids.append(assessment.publisher_id)

        # every publisher should have been scored; a few omissions are filled in later
        missing = []
        for publisher in self.catalog.publishers:
            if publisher.id not in seen_ids:
                missing.append(publisher.id)
        if len(missing) > self.config.max_missing_before_retry:
            errors.append("missing assessments for publishers: " + ", ".join(missing))

        # the same publisher twice means the model lost track of the list
        duplicates = []
        for publisher_id in seen_ids:
            if seen_ids.count(publisher_id) > 1 and publisher_id not in duplicates:
                duplicates.append(publisher_id)
        if duplicates:
            errors.append("duplicate publisher ids: " + ", ".join(sorted(duplicates)))

        # anything not recommended must say why: the exclusion list is a deliverable
        for assessment in output.assessments:
            has_reason = bool((assessment.exclusion_reason or "").strip())
            if assessment.verdict is not Verdict.RECOMMEND and not has_reason:
                errors.append(f"{assessment.publisher_id}: exclusion_reason is required for verdict "
                              f"'{assessment.verdict}'")
        return errors

    # ---- enrichment + ranking ----

    def apply(self, output: MatchOutput, signals: dict[str, FitSignals],
              brief: AdvertiserBrief) -> list[PublisherAssessment]:
        """Apply every rule, rank, bound the recommended set, and return all 20 publishers."""
        drafts_by_id = {}
        for assessment in output.assessments:
            drafts_by_id[assessment.publisher_id] = assessment

        # --- Step 1: one row per catalog publisher, with the per-publisher rules applied ---
        rows = []
        for publisher in self.catalog.publishers:
            draft = drafts_by_id.get(publisher.id)
            rows.append(self._apply_row_rules(draft, publisher.id, signals[publisher.id], brief))

        # --- Step 2: rank, bound the recommended set, rank again (verdicts may have changed) ---
        self._rank(rows)
        self._bound_recommended(rows, brief.is_off_catalog)
        self._rank(rows)

        # --- Step 3: turn rows into the API shape, numbered by rank ---
        assessments = []
        for rank, row in enumerate(rows, start=1):
            assessments.append(self._to_assessment(row, rank))
        return assessments

    def _apply_row_rules(self, draft: PublisherAssessmentDraft | None, publisher_id: str,
                         signals: FitSignals, brief: AdvertiserBrief) -> _Row:
        cfg = self.config
        tags = []

        # Rule: a publisher the model skipped is excluded by default, and tagged as filled in.
        if draft is None:
            draft = PublisherAssessmentDraft(
                publisher_id=publisher_id, verdict=Verdict.EXCLUDE, score=0,
                subscores=Subscores(audience_fit=0, category_fit=0, price_fit=0, context_fit=0),
                reasons=[], concerns=[], exclusion_reason=NOT_ASSESSED_REASON,
            )
            tags.append(GuardrailTag.FILLED_MISSING)

        score = clamp_score(draft.score)
        verdict = draft.verdict

        # Rule: an off-catalog advertiser (B2B software, outdoor gear...) can be explored
        # but never recommended, and no publisher may look like a strong fit.
        if brief.is_off_catalog:
            if score > cfg.off_catalog_cap:
                score = cfg.off_catalog_cap
                tags.append(GuardrailTag.OFF_CATALOG_CAP)
            if verdict is Verdict.RECOMMEND:
                verdict = Verdict.CONSIDER
                tags.append(GuardrailTag.OFF_CATALOG_NO_RECOMMEND)

        # Rule: a $600 product on a $40-basket publisher is a mismatch, unless the shoppers
        # are high income, or the product is a gift (the buyer is not the wearer).
        publisher = self.catalog.publisher(publisher_id)
        price_far_above_basket = signals.aov_ratio > cfg.aov_mismatch_ratio
        shoppers_are_high_income = publisher.audience.income_tier is IncomeTier.HIGH
        is_gifting = brief.purchase_model is PurchaseModel.GIFTING
        if price_far_above_basket and not shoppers_are_high_income and not is_gifting:
            if score > cfg.aov_mismatch_cap:
                score = cfg.aov_mismatch_cap
                tags.append(GuardrailTag.AOV_MISMATCH_CAP)

        # Rule: "recommend" with a low score is a contradiction; downgrade it.
        if verdict is Verdict.RECOMMEND and score < cfg.promote_min_score:
            verdict = Verdict.CONSIDER
            tags.append(GuardrailTag.VERDICT_SCORE_MISMATCH)

        # Rule: a score far from the computed prior is not overridden, but it is flagged
        # so a reviewer looks at the reasons.
        if abs(score - signals.prior) > cfg.divergence_from_prior:
            tags.append(GuardrailTag.DIVERGES_FROM_SIGNALS)

        return _Row(draft=draft, signals=signals, score=score, verdict=verdict, tags=tags)

    @staticmethod
    def _rank(rows: list[_Row]) -> None:
        """Best first: by verdict, then score, then reach as the tie-breaker."""

        def sort_key(row: _Row):
            return (VERDICT_ORDER[row.verdict], -row.score, -row.signals.reach_index)

        rows.sort(key=sort_key)

    def _bound_recommended(self, rows: list[_Row], off_catalog: bool) -> None:
        """Keep the recommended set between min and max publishers (rows must be ranked)."""
        cfg = self.config

        # the model's recommendations that clear the bar, best first, at most `max_recommended`
        recommended = []
        for row in rows:
            if row.verdict is Verdict.RECOMMEND and row.score >= cfg.recommend_min_score:
                recommended.append(row)
        recommended = recommended[:cfg.max_recommended]

        # too few: promote the best "consider" rows above the promotion bar (never off-catalog)
        minimum = min(cfg.min_recommended, cfg.max_recommended)
        if not off_catalog and len(recommended) < minimum:
            for row in rows:
                if len(recommended) >= minimum:
                    break
                already_in = row in recommended
                promotable = row.verdict is not Verdict.EXCLUDE and row.score >= cfg.promote_min_score
                if not already_in and promotable:
                    row.verdict = Verdict.RECOMMEND
                    row.tags.append(GuardrailTag.PROMOTED_TO_FILL_MINIMUM)
                    recommended.append(row)

        # too many, or below the bar: everything else marked "recommend" becomes "consider"
        for row in rows:
            if row.verdict is Verdict.RECOMMEND and row not in recommended:
                row.verdict = Verdict.CONSIDER
                row.tags.append(GuardrailTag.BEYOND_TOP_N)

    def _to_assessment(self, row: _Row, rank: int) -> PublisherAssessment:
        draft = row.draft

        # a non-recommended publisher always carries a reason, even if the verdict changed here
        exclusion_reason = None
        if row.verdict is not Verdict.RECOMMEND:
            exclusion_reason = draft.exclusion_reason or BEYOND_TOP_SET_REASON

        subscores = draft.subscores
        return PublisherAssessment(
            publisher_id=draft.publisher_id,
            publisher_name=self.catalog.publisher(draft.publisher_id).name,
            verdict=row.verdict,
            score=row.score,
            subscores=Subscores(
                audience_fit=clamp_score(subscores.audience_fit),
                category_fit=clamp_score(subscores.category_fit),
                price_fit=clamp_score(subscores.price_fit),
                context_fit=clamp_score(subscores.context_fit),
            ),
            reasons=draft.reasons,
            concerns=draft.concerns,
            exclusion_reason=exclusion_reason,
            signals=row.signals,
            guardrails_applied=row.tags,
            rank=rank,
        )


def clamp_score(value: float) -> float:
    """Whole number inside 0-100."""
    return float(max(0, min(100, round(value))))
