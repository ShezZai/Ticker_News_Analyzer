"""Offline unit tests for the E2E pipeline eval. No DB, no network."""

from ticker_news.evals.pipeline_eval import score_directional


class TestScoreDirectional:
    def test_buy_and_price_up_agrees(self):
        value, comment = score_directional("buy", 2.5)
        assert value == 1.0
        assert "agree" in comment

    def test_buy_and_price_down_disagrees(self):
        value, comment = score_directional("buy", -1.2)
        assert value == 0.0
        assert "disagree" in comment

    def test_buy_and_flat_price_disagrees(self):
        value, _ = score_directional("buy", 0.0)
        assert value == 0.0

    def test_sell_and_price_down_agrees(self):
        value, _ = score_directional("sell", -3.0)
        assert value == 1.0

    def test_sell_and_price_up_disagrees(self):
        value, _ = score_directional("sell", 1.7)
        assert value == 0.0

    def test_hold_is_excluded(self):
        value, comment = score_directional("hold", 2.0)
        assert value is None
        assert "hold" in comment

    def test_no_verdict_is_excluded_with_reason(self):
        value, comment = score_directional(None, None, skip_reason="category=recap/review")
        assert value is None
        assert "category=recap/review" in comment

    def test_missing_price_data_is_excluded(self):
        value, comment = score_directional("buy", None, skip_reason="no tradeable entry/exit bar")
        assert value is None
        assert "no price data" in comment

    def test_unknown_action_is_excluded(self):
        value, comment = score_directional("strong-buy", 1.0)
        assert value is None
        assert "strong-buy" in comment
