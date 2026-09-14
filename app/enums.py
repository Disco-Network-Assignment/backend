"""Domain vocabularies as string enums - one definition, no scattered literals.
StrEnum members ARE strings, so they serialize as their raw values in JSON and in the
structured-output schemas the agents fill in."""

from enum import StrEnum


class InputQuality(StrEnum):
    CLEAR = "clear"                # product and audience/price stated or safely inferable
    VAGUE = "vague"                # a real product is implied, key facts are missing
    AMBIGUOUS = "ambiguous"        # several materially different readings
    OFF_CATALOG = "off_catalog"    # a real business this consumer catalog cannot serve
    INSUFFICIENT = "insufficient"  # no usable business information at all


class PriceTier(StrEnum):
    BUDGET = "budget"
    MID = "mid"
    PREMIUM = "premium"
    LUXURY = "luxury"


class PurchaseModel(StrEnum):
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    GIFTING = "gifting"
    B2B = "b2b"


class GenderSkew(StrEnum):
    FEMALE = "female"
    MALE = "male"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


class IncomeTier(StrEnum):
    """The three tiers the publisher catalog uses; the brief adds `None` for unknown."""

    MID = "mid"
    MID_HIGH = "mid-high"
    HIGH = "high"


class ProductCategory(StrEnum):
    """Controlled vocabulary the intake agent maps every advertiser onto. The taxonomy
    (domain/categories.py) bridges each value to publisher subcategories and persona
    affinities, which is what makes deterministic category signals possible."""

    PET_FOOD = "pet_food"
    PET_SUPPLIES = "pet_supplies"
    PET_HEALTH = "pet_health"
    ACTIVEWEAR = "activewear"
    WOMENS_APPAREL = "womens_apparel"
    MENS_APPAREL = "mens_apparel"
    PLUS_SIZE_APPAREL = "plus_size_apparel"
    FOOTWEAR = "footwear"
    BASICS_SOCKS_UNDERWEAR = "basics_socks_underwear"
    LUXURY_ACCESSORIES = "luxury_accessories"
    BEAUTY_SKINCARE = "beauty_skincare"
    HAIRCARE = "haircare"
    SUPPLEMENTS_VITAMINS = "supplements_vitamins"
    FUNCTIONAL_BEVERAGES = "functional_beverages"
    ALCOHOL = "alcohol"
    SNACKS_PROTEIN = "snacks_protein"
    GROCERIES_PANTRY = "groceries_pantry"
    MEAL_KITS = "meal_kits"
    HOUSEHOLD_CLEANING = "household_cleaning"
    HOME_TEXTILES_BEDDING = "home_textiles_bedding"
    KITCHEN_COOKWARE = "kitchen_cookware"
    HOME_DECOR_CANDLES = "home_decor_candles"
    GIFTS = "gifts"
    KIDS_BABY = "kids_baby"
    FITNESS_SERVICES = "fitness_services"
    WELLNESS_SERVICES = "wellness_services"
    OUTDOOR_GEAR = "outdoor_gear"
    B2B_SOFTWARE = "b2b_software"
    OTHER = "other"


class BrandAttribute(StrEnum):
    SUSTAINABLE = "sustainable"
    PREMIUM = "premium"
    LUXURY = "luxury"
    VALUE = "value"
    SCIENCE_BACKED = "science_backed"
    SUBSCRIPTION = "subscription"
    GIFTING = "gifting"
    PLAYFUL = "playful"
    HERITAGE = "heritage"
    CONVENIENCE = "convenience"
    INCLUSIVE = "inclusive"
    NATURAL_CLEAN = "natural_clean"
    PERSONALIZED = "personalized"
    PERFORMANCE = "performance"


class Verdict(StrEnum):
    RECOMMEND = "recommend"  # would put budget here
    CONSIDER = "consider"    # a plausible small test
    EXCLUDE = "exclude"      # would not run


class GuardrailTag(StrEnum):
    """Every rule the guard applies to a model verdict leaves a tag the UI can show."""

    FILLED_MISSING = "filled_missing"
    OFF_CATALOG_CAP = "off_catalog_cap"
    OFF_CATALOG_NO_RECOMMEND = "off_catalog_no_recommend"
    AOV_MISMATCH_CAP = "aov_mismatch_cap"
    VERDICT_SCORE_MISMATCH = "verdict_score_mismatch"
    DIVERGES_FROM_SIGNALS = "diverges_from_signals"
    PROMOTED_TO_FILL_MINIMUM = "promoted_to_fill_minimum"
    BEYOND_TOP_N = "beyond_top_n"


class LintSeverity(StrEnum):
    HARD = "hard"  # regenerate once, then show with a warning badge
    SOFT = "soft"  # shown as a warning, never blocks


class RouteFlag(StrEnum):
    """What the router asks the UI to surface for a run that continues with caveats."""

    ASSUMPTIONS = "assumptions"
    INTERPRETATIONS = "interpretations"
    OFF_CATALOG = "off_catalog"


class ConfigStatus(StrEnum):
    DRAFT = "draft"
    NOT_RECOMMENDED = "not_recommended"


class Objective(StrEnum):
    ACQUISITION = "acquisition"
    CONSIDERATION = "consideration"
    SEASONAL_GIFTING = "seasonal_gifting"
    RETENTION = "retention"


class BidModel(StrEnum):
    CPM = "CPM"
    CPC = "CPC"
    CPA = "CPA"


class Stage(StrEnum):
    INTAKE = "intake"
    SIGNALS = "signals"
    MATCH = "match"
    PERSONAS = "personas"
    CREATIVE = "creative"
    CONFIG = "config"
    SUMMARY = "summary"
    DONE = "done"
    STOPPED = "stopped"
    ERROR = "error"


class EventStatus(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureKind(StrEnum):
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    REFUSAL = "refusal"
    API = "api"
    UNKNOWN = "unknown"
