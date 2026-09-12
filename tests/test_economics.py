from app.domain.economics import DEFAULT_ECONOMICS
from app.enums import IncomeTier, PriceTier, PurchaseModel


class TestEconomics:
    def test_cpm_band_scales_with_fit(self):
        assert DEFAULT_ECONOMICS.cpm_band(IncomeTier.HIGH, 90) == (20.7, 32.2)
        assert DEFAULT_ECONOMICS.cpm_band(IncomeTier.MID, 60) == (6.8, 10.2)

    def test_pilot_tiers(self):
        assert DEFAULT_ECONOMICS.pilot_for(0.9).total_usd == 5000
        assert DEFAULT_ECONOMICS.pilot_for(0.6).total_usd == 3000
        assert DEFAULT_ECONOMICS.pilot_for(0.1).flight_days == 7

    def test_target_cpa_by_purchase_model(self):
        assert DEFAULT_ECONOMICS.target_cpa(100, PurchaseModel.ONE_TIME) == 30
        assert DEFAULT_ECONOMICS.target_cpa(100, PurchaseModel.SUBSCRIPTION) == 60
        assert DEFAULT_ECONOMICS.target_cpa(100, PurchaseModel.B2B) is None

    def test_price_point_falls_back_to_tier(self):
        assert DEFAULT_ECONOMICS.price_point(None, PriceTier.LUXURY) == 600
        assert DEFAULT_ECONOMICS.price_point(42, PriceTier.LUXURY) == 42

    def test_forecast_maths(self):
        forecast = DEFAULT_ECONOMICS.forecast(5000, 12.5)
        assert forecast.impressions == 400_000
        assert forecast.clicks == 3600
        assert forecast.conversions == 108
        assert forecast.cpa_usd == 46.3

    def test_zero_budget_forecast(self):
        assert DEFAULT_ECONOMICS.forecast(0, 12).conversions == 0

    def test_describe_lists_every_assumption_family(self):
        text = " ".join(DEFAULT_ECONOMICS.describe())
        assert "CPM" in text and "CTR" in text and "CPA" in text and "Pilot" in text
