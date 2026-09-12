"""Rules a model verdict must never break, applied after the matcher agent has scored the
catalog: every publisher accounted for exactly once, scores clamped, off-catalog briefs capped,
absurd price mismatches capped, verdicts consistent with scores, a bounded recommended set.
Every rule that fires leaves a GuardrailTag on the assessment so the UI can show it."""

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

_VERDICT_ORDER = {Verdict.RECOMMEND: 0, Verdict.CONSIDER: 1, Verdict.EXCLUDE: 2}


@dataclass(frozen=True)
class GuardConfig:
    recommend_min_score: float = 70.0   # below this a "recommend" becomes "consider"
    promote_min_score: float = 60.0     # a "consider" can be promoted to fill the minimum
    max_recommended: int = 6
    min_recommended: int = 3
    off_catalog_cap: float = 40.0
    aov_mismatch_ratio: float = 6.0     # price more than 6x the publisher's AOV...
    aov_mismatch_cap: float = 55.0      # ...caps the score, except on high-income publishers
    divergence_from_prior: float = 40.0
    max_missing_before_retry: int = 3   # more omissions than this -> ask the model again


@dataclass
class _Row:
    draft: PublisherAssessmentDraft
    signals: FitSignals
    score: float
    verdict: Verdict
    tags: list[GuardrailTag] = field(default_factory=list)


class AssessmentGuard:
    def __init__(self, catalog: CatalogRepository, config: GuardConfig = GuardConfig()) -> None:
        self._catalog = catalog
        self._config = config

    # ---- validation (drives the retry loop) ----
    def validation_errors(self, output: MatchOutput) -> list[str]:
        ids = [a.publisher_id for a in output.assessments]
        errors: list[str] = []
        missing = [p.id for p in self._catalog.publishers if p.id not in ids]
        if len(missing) > self._config.max_missing_before_retry:
            errors.append(f"missing assessments for publishers: {', '.join(missing)}")
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            errors.append(f"duplicate publisher ids: {', '.join(duplicates)}")
        for a in output.assessments:
            if a.verdict is not Verdict.RECOMMEND and not (a.exclusion_reason or "").strip():
                errors.append(f"{a.publisher_id}: exclusion_reason is required for verdict "
                              f"'{a.verdict}'")
        return errors

    # ---- enrichment + ranking ----
    def apply(self, output: MatchOutput, signals: dict[str, FitSignals],
              brief: AdvertiserBrief) -> list[PublisherAssessment]:
        drafts = {a.publisher_id: a for a in output.assessments}
        rows = [self._row(drafts.get(p.id), p.id, signals[p.id], brief)
                for p in self._catalog.publishers]
        self._rank(rows)
        self._bound_recommended(rows, brief.is_off_catalog)
        self._rank(rows)
        return [self._assessment(row, rank) for rank, row in enumerate(rows, start=1)]

    def _row(self, draft: PublisherAssessmentDraft | None, publisher_id: str,
             signals: FitSignals, brief: AdvertiserBrief) -> _Row:
        cfg = self._config
        tags: list[GuardrailTag] = []
        if draft is None:
            draft = PublisherAssessmentDraft(
                publisher_id=publisher_id, verdict=Verdict.EXCLUDE, score=0,
                subscores=Subscores(audience_fit=0, category_fit=0, price_fit=0, context_fit=0),
                reasons=[], concerns=[],
                exclusion_reason="Not assessed by the model; excluded by default.",
            )
            tags.append(GuardrailTag.FILLED_MISSING)
        score = _clamp(draft.score)
        verdict = draft.verdict

        if brief.is_off_catalog:
            if score > cfg.off_catalog_cap:
                score = cfg.off_catalog_cap
                tags.append(GuardrailTag.OFF_CATALOG_CAP)
            if verdict is Verdict.RECOMMEND:
                verdict = Verdict.CONSIDER
                tags.append(GuardrailTag.OFF_CATALOG_NO_RECOMMEND)

        publisher = self._catalog.publisher(publisher_id)
        price_mismatch = (signals.aov_ratio > cfg.aov_mismatch_ratio
                          and publisher.audience.income_tier is not IncomeTier.HIGH
                          and brief.purchase_model is not PurchaseModel.GIFTING)
        if price_mismatch and score > cfg.aov_mismatch_cap:
            score = cfg.aov_mismatch_cap
            tags.append(GuardrailTag.AOV_MISMATCH_CAP)

        if verdict is Verdict.RECOMMEND and score < cfg.promote_min_score:
            verdict = Verdict.CONSIDER
            tags.append(GuardrailTag.VERDICT_SCORE_MISMATCH)

        if abs(score - signals.prior) > cfg.divergence_from_prior:
            tags.append(GuardrailTag.DIVERGES_FROM_SIGNALS)

        return _Row(draft=draft, signals=signals, score=score, verdict=verdict, tags=tags)

    @staticmethod
    def _rank(rows: list[_Row]) -> None:
        rows.sort(key=lambda r: (_VERDICT_ORDER[r.verdict], -r.score, -r.signals.reach_index))

    def _bound_recommended(self, rows: list[_Row], off_catalog: bool) -> None:
        cfg = self._config
        recommended = [r for r in rows
                       if r.verdict is Verdict.RECOMMEND and r.score >= cfg.recommend_min_score]
        recommended = recommended[: cfg.max_recommended]
        minimum = min(cfg.min_recommended, cfg.max_recommended)
        if not off_catalog and len(recommended) < minimum:
            for row in rows:
                if len(recommended) >= minimum:
                    break
                if row not in recommended and row.verdict is not Verdict.EXCLUDE \
                        and row.score >= cfg.promote_min_score:
                    row.verdict = Verdict.RECOMMEND
                    row.tags.append(GuardrailTag.PROMOTED_TO_FILL_MINIMUM)
                    recommended.append(row)
        for row in rows:
            if row.verdict is Verdict.RECOMMEND and row not in recommended:
                row.verdict = Verdict.CONSIDER
                row.tags.append(GuardrailTag.BEYOND_TOP_N)

    def _assessment(self, row: _Row, rank: int) -> PublisherAssessment:
        d = row.draft
        exclusion = d.exclusion_reason
        if row.verdict is not Verdict.RECOMMEND and not exclusion:
            exclusion = "Not in the recommended set after ranking."
        return PublisherAssessment(
            publisher_id=d.publisher_id,
            publisher_name=self._catalog.publisher(d.publisher_id).name,
            verdict=row.verdict,
            score=row.score,
            subscores=Subscores(**{k: _clamp(v) for k, v in d.subscores.model_dump().items()}),
            reasons=d.reasons,
            concerns=d.concerns,
            exclusion_reason=exclusion if row.verdict is not Verdict.RECOMMEND else None,
            signals=row.signals,
            guardrails_applied=row.tags,
            rank=rank,
        )


def _clamp(value: float) -> float:
    return float(max(0, min(100, round(value))))
