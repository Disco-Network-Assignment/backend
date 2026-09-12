"""Expectations for the 15 sample advertisers - the catalog is small enough that a reviewer can
say which publishers must (not) appear and which personas are plausible. Expectations are
"any of" wherever the model is legitimately free to choose; "must not" lists are the real
regression guard."""

from dataclasses import dataclass

from app.enums import ConfigStatus, InputQuality


@dataclass(frozen=True)
class EvalCase:
    number: int
    slug: str
    quality: tuple[InputQuality, ...]
    stopped: bool = False
    top3_any: tuple[str, ...] = ()          # at least one of these in the top-3 recommended
    must_not_recommend: tuple[str, ...] = ()
    personas_any: tuple[str, ...] = ()
    personas_none: tuple[str, ...] = ()
    max_personas: int | None = None
    status: ConfigStatus | None = None
    max_budget_usd: float | None = None
    needs_interpretations: bool = False


PET = ("pub_007", "pub_009", "pub_018")
BEAUTY = ("pub_013", "pub_019")
MATURE_APPAREL = ("pub_004", "pub_005")

CASES: tuple[EvalCase, ...] = (
    EvalCase(1, "senior-dog-food", (InputQuality.CLEAR,), top3_any=("pub_007", "pub_009"),
             must_not_recommend=(*BEAUTY, "pub_005", "pub_003", "pub_001"),
             personas_any=("persona_004",), personas_none=("persona_010",), status=ConfigStatus.DRAFT),
    EvalCase(2, "sustainable-activewear", (InputQuality.CLEAR,),
             top3_any=("pub_002", "pub_017", "pub_016"), must_not_recommend=(*PET, "pub_001", "pub_005"),
             personas_any=("persona_006", "persona_009"), status=ConfigStatus.DRAFT),
    EvalCase(3, "adaptogen-drink", (InputQuality.CLEAR,), top3_any=("pub_020", "pub_008", "pub_001"),
             must_not_recommend=(*PET, "pub_005", "pub_011"),
             personas_any=("persona_003", "persona_001", "persona_009", "persona_007"),
             status=ConfigStatus.DRAFT),
    EvalCase(4, "vermont-candles", (InputQuality.CLEAR,), top3_any=("pub_014", "pub_011", "pub_010"),
             must_not_recommend=(*PET, "pub_002", "pub_012", "pub_015"),
             personas_any=("persona_010", "persona_005", "persona_003"), status=ConfigStatus.DRAFT),
    EvalCase(5, "feel-better", (InputQuality.VAGUE, InputQuality.AMBIGUOUS), max_budget_usd=1500,
             max_personas=3),
    EvalCase(6, "ski-shells", (InputQuality.OFF_CATALOG,),
             must_not_recommend=("pub_001", "pub_018", "pub_013", "pub_012", "pub_019")),
    EvalCase(7, "dental-saas", (InputQuality.OFF_CATALOG,), status=ConfigStatus.NOT_RECOMMENDED,
             must_not_recommend=tuple(f"pub_{i:03d}" for i in range(1, 21))),
    EvalCase(8, "thing-for-moms", (InputQuality.AMBIGUOUS, InputQuality.VAGUE), max_personas=3,
             max_budget_usd=1500, needs_interpretations=True),
    EvalCase(9, "refillable-cleaning", (InputQuality.CLEAR,),
             top3_any=("pub_008", "pub_014", "pub_016", "pub_017", "pub_001", "pub_015"),
             must_not_recommend=(*BEAUTY, *PET, "pub_005"), personas_any=("persona_006",),
             status=ConfigStatus.DRAFT),
    EvalCase(10, "italian-handbags", (InputQuality.CLEAR,), top3_any=("pub_005", "pub_004", "pub_016"),
             must_not_recommend=("pub_001", "pub_018", "pub_013", "pub_012", "pub_009", "pub_019"),
             personas_any=("persona_005",), status=ConfigStatus.DRAFT),
    EvalCase(11, "protein-bars", (InputQuality.CLEAR,),
             top3_any=("pub_002", "pub_003", "pub_020", "pub_008", "pub_001", "pub_015"),
             must_not_recommend=(*MATURE_APPAREL, *PET, "pub_013"), status=ConfigStatus.DRAFT),
    EvalCase(12, "new-cat-owners", (InputQuality.CLEAR,), top3_any=PET,
             must_not_recommend=("pub_002", *MATURE_APPAREL, "pub_011", "pub_013"),
             personas_any=("persona_004",), status=ConfigStatus.DRAFT),
    EvalCase(13, "value-supplements", (InputQuality.CLEAR,),
             top3_any=("pub_002", "pub_001", "pub_020", "pub_017", "pub_012", "pub_003"),
             must_not_recommend=(*MATURE_APPAREL, *PET, "pub_011"),
             personas_any=("persona_009", "persona_008"), status=ConfigStatus.DRAFT),
    EvalCase(14, "linen-bedding", (InputQuality.CLEAR,), top3_any=("pub_011", "pub_014", "pub_004", "pub_016"),
             must_not_recommend=("pub_001", "pub_018", "pub_013", "pub_007", "pub_009"),
             personas_any=("persona_005", "persona_006", "persona_001", "persona_010"),
             status=ConfigStatus.DRAFT),
    EvalCase(15, "idk", (InputQuality.INSUFFICIENT,), stopped=True),
)
