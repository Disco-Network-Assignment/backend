"""Bridge between the intake vocabulary (ProductCategory) and the two vocabularies in the
data pack: publisher `category`/`subcategories` and persona `category_affinities`. They do
not match one-to-one (`pet_food` vs `pet_health`, `activewear` vs `sustainable_apparel`),
so every deterministic category signal goes through this table.

`direct` = the same shelf; `adjacent` = a complementary shelf a shopper would accept next to
what they just bought; `persona` = affinity terms from shopper_personas.json."""

from dataclasses import dataclass

from app.enums import ProductCategory


@dataclass(frozen=True)
class CategoryTerms:
    direct: tuple[str, ...]
    adjacent: tuple[str, ...]
    persona: tuple[str, ...]


TAXONOMY: dict[ProductCategory, CategoryTerms] = {
    ProductCategory.PET_FOOD: CategoryTerms(
        direct=("pet", "pet_food"),
        adjacent=("pet_supplies", "pet_pharmacy"),
        persona=("pet_food", "pet_health", "subscription_boxes"),
    ),
    ProductCategory.PET_SUPPLIES: CategoryTerms(
        direct=("pet", "pet_supplies", "toys", "treats"),
        adjacent=("pet_food",),
        persona=("pet_supplies", "pet_food", "subscription_boxes"),
    ),
    ProductCategory.PET_HEALTH: CategoryTerms(
        direct=("pet_pharmacy", "pet"),
        adjacent=("pet_food", "pet_supplies"),
        persona=("pet_health",),
    ),
    ProductCategory.ACTIVEWEAR: CategoryTerms(
        direct=("activewear",),
        adjacent=("apparel", "fitness_classes", "yoga", "personal_training", "shoes"),
        persona=("activewear", "fitness", "recovery", "sustainable_apparel"),
    ),
    ProductCategory.WOMENS_APPAREL: CategoryTerms(
        direct=("apparel", "women", "casual", "workwear", "classic", "mid-life"),
        adjacent=("shoes", "intimates", "plus_size"),
        persona=("apparel", "fashion", "classic_apparel"),
    ),
    ProductCategory.MENS_APPAREL: CategoryTerms(
        direct=("men",),
        adjacent=("apparel", "activewear", "basics"),
        persona=("apparel", "fashion"),
    ),
    ProductCategory.PLUS_SIZE_APPAREL: CategoryTerms(
        direct=("plus_size",),
        adjacent=("women", "apparel", "intimates", "workwear"),
        persona=("apparel", "fashion"),
    ),
    ProductCategory.FOOTWEAR: CategoryTerms(
        direct=("shoes",),
        adjacent=("activewear", "apparel", "sustainable"),
        persona=("apparel", "fashion", "activewear"),
    ),
    ProductCategory.BASICS_SOCKS_UNDERWEAR: CategoryTerms(
        direct=("socks", "underwear", "basics", "intimates"),
        adjacent=("gifting", "apparel"),
        persona=("premium_basics", "apparel"),
    ),
    ProductCategory.LUXURY_ACCESSORIES: CategoryTerms(
        direct=(),
        adjacent=("women", "mid-life", "classic", "workwear", "shoes"),
        persona=("classic_apparel", "fashion"),
    ),
    ProductCategory.BEAUTY_SKINCARE: CategoryTerms(
        direct=("beauty", "skincare", "makeup"),
        adjacent=("haircare", "dtc", "spa"),
        persona=("beauty", "clean_beauty"),
    ),
    ProductCategory.HAIRCARE: CategoryTerms(
        direct=("haircare",),
        adjacent=("beauty", "skincare", "personalized"),
        persona=("beauty", "clean_beauty"),
    ),
    ProductCategory.SUPPLEMENTS_VITAMINS: CategoryTerms(
        direct=("vitamins", "supplements", "wellness_dtc"),
        adjacent=("fitness_classes", "activewear", "organic", "natural"),
        persona=("supplements", "wellness", "fitness"),
    ),
    ProductCategory.FUNCTIONAL_BEVERAGES: CategoryTerms(
        direct=("functional_beverages", "soda_alternative", "gut_health", "beverages"),
        adjacent=("organic", "natural", "alcohol", "convenience", "groceries"),
        persona=("functional_beverages", "wellness"),
    ),
    ProductCategory.ALCOHOL: CategoryTerms(
        direct=("alcohol",),
        adjacent=("convenience", "beverages"),
        persona=("convenience",),
    ),
    ProductCategory.SNACKS_PROTEIN: CategoryTerms(
        direct=("pantry", "natural"),
        adjacent=("groceries", "functional_beverages", "fitness_classes", "activewear",
                  "convenience", "meal_kits"),
        persona=("supplements", "fitness", "groceries", "convenience"),
    ),
    ProductCategory.GROCERIES_PANTRY: CategoryTerms(
        direct=("groceries", "pantry", "organic", "natural"),
        adjacent=("meal_kits", "convenience"),
        persona=("organic_grocery", "premium_grocery", "groceries"),
    ),
    ProductCategory.MEAL_KITS: CategoryTerms(
        direct=("meal_kits",),
        adjacent=("groceries", "convenience"),
        persona=("meal_kits", "convenience"),
    ),
    ProductCategory.HOUSEHOLD_CLEANING: CategoryTerms(
        direct=("household",),
        adjacent=("non_toxic", "kitchen", "groceries", "organic"),
        persona=("household", "refillable_products", "family_products"),
    ),
    ProductCategory.HOME_TEXTILES_BEDDING: CategoryTerms(
        direct=("bedding", "bath", "home_textiles"),
        adjacent=("home", "cookware", "kitchen"),
        persona=("home_goods", "home_decor"),
    ),
    ProductCategory.KITCHEN_COOKWARE: CategoryTerms(
        direct=("cookware", "kitchen", "non_toxic"),
        adjacent=("home", "bedding", "meal_kits", "groceries"),
        persona=("home_goods",),
    ),
    ProductCategory.HOME_DECOR_CANDLES: CategoryTerms(
        direct=("home", "home_textiles"),
        adjacent=("gifting", "cookware", "bedding", "natural"),
        persona=("home_decor", "small_batch", "home_goods"),
    ),
    ProductCategory.GIFTS: CategoryTerms(
        direct=("gifting",),
        adjacent=("home", "beauty", "basics", "socks"),
        persona=("gourmet_food", "premium_basics", "home_goods", "beauty"),
    ),
    ProductCategory.KIDS_BABY: CategoryTerms(
        direct=(),
        adjacent=("meal_kits", "household", "groceries"),
        persona=("kids_products", "family_products"),
    ),
    ProductCategory.FITNESS_SERVICES: CategoryTerms(
        direct=("fitness_classes", "yoga", "personal_training", "wellness_services"),
        adjacent=("activewear", "spa"),
        persona=("fitness_services", "fitness"),
    ),
    ProductCategory.WELLNESS_SERVICES: CategoryTerms(
        direct=("spa", "wellness_services", "yoga"),
        adjacent=("vitamins", "supplements"),
        persona=("wellness",),
    ),
    ProductCategory.OUTDOOR_GEAR: CategoryTerms(
        direct=(),
        adjacent=("activewear", "shoes"),
        persona=("activewear", "fitness"),
    ),
    ProductCategory.B2B_SOFTWARE: CategoryTerms(direct=(), adjacent=(), persona=()),
    ProductCategory.OTHER: CategoryTerms(direct=(), adjacent=(), persona=()),
}


def terms_for(category: ProductCategory) -> CategoryTerms:
    return TAXONOMY[category]
