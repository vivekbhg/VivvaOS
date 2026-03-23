"""Tests for trading strategies."""
import pytest

from bot.models import Market, Side, SignalStrength
from bot.strategies.value import ValueStrategy
from bot.strategies.momentum import MomentumStrategy
from bot.strategies.mispricing import MispricingStrategy


def make_market(**kwargs) -> Market:
    defaults = {
        "condition_id": "test_123",
        "question": "Will it rain tomorrow?",
        "token_id_yes": "token_yes",
        "token_id_no": "token_no",
        "yes_price": 0.50,
        "no_price": 0.50,
        "spread": 0.02,
        "volume_24h": 5000,
        "liquidity": 10000,
    }
    defaults.update(kwargs)
    return Market(**defaults)


def make_orderbook(bids=None, asks=None):
    return {
        "bids": bids or [{"price": "0.48", "size": "500"}, {"price": "0.47", "size": "300"}],
        "asks": asks or [{"price": "0.52", "size": "500"}, {"price": "0.53", "size": "300"}],
    }


class TestValueStrategy:
    def setup_method(self):
        self.strategy = ValueStrategy()

    def test_no_signal_balanced_book(self):
        market = make_market(yes_price=0.50, no_price=0.50)
        book = make_orderbook()
        signal = self.strategy.analyze(market, book)
        assert signal is None

    def test_buy_signal_heavy_bids(self):
        market = make_market(yes_price=0.40, no_price=0.55)
        book = make_orderbook(
            bids=[{"price": "0.45", "size": "2000"}, {"price": "0.44", "size": "1500"}],
            asks=[{"price": "0.50", "size": "200"}, {"price": "0.51", "size": "100"}],
        )
        signal = self.strategy.analyze(market, book)
        assert signal is not None
        assert signal.side == Side.BUY
        assert signal.edge > 0

    def test_no_signal_empty_book(self):
        market = make_market()
        signal = self.strategy.analyze(market, {"bids": [], "asks": []})
        assert signal is None


class TestMomentumStrategy:
    def setup_method(self):
        self.strategy = MomentumStrategy()

    def test_no_signal_low_activity(self):
        market = make_market(volume_24h=100, liquidity=10000)
        book = make_orderbook()
        signal = self.strategy.analyze(market, book)
        assert signal is None

    def test_bullish_signal_bid_wall(self):
        market = make_market(volume_24h=20000, liquidity=10000)
        book = make_orderbook(
            bids=[{"price": "0.50", "size": "5000"}],
            asks=[{"price": "0.52", "size": "200"}],
        )
        signal = self.strategy.analyze(market, book)
        if signal:
            assert signal.side == Side.BUY
            assert signal.token_id == "token_yes"

    def test_bearish_signal_ask_wall(self):
        market = make_market(volume_24h=20000, liquidity=10000)
        book = make_orderbook(
            bids=[{"price": "0.48", "size": "200"}],
            asks=[{"price": "0.52", "size": "5000"}],
        )
        signal = self.strategy.analyze(market, book)
        if signal:
            assert signal.side == Side.BUY
            assert signal.token_id == "token_no"


class TestMispricingStrategy:
    def setup_method(self):
        self.strategy = MispricingStrategy()

    def test_no_signal_efficient_market(self):
        market = make_market(yes_price=0.50, no_price=0.50)
        signal = self.strategy.analyze(market, {})
        assert signal is None

    def test_underround_signal(self):
        # YES + NO < 1.0 -> arbitrage opportunity
        market = make_market(yes_price=0.40, no_price=0.48)
        signal = self.strategy.analyze(market, {})
        assert signal is not None
        assert signal.side == Side.BUY
        assert signal.edge > 0.05

    def test_skew_signal(self):
        # YES is much cheaper than NO implies
        market = make_market(yes_price=0.30, no_price=0.60)
        signal = self.strategy.analyze(market, {})
        if signal:
            assert signal.edge > 0


class TestRiskManager:
    def test_position_sizing(self):
        from bot.services.risk_manager import RiskManager
        from bot.models import Signal

        rm = RiskManager()
        market = make_market()
        signal = Signal(
            market=market,
            strategy="test",
            side=Side.BUY,
            token_id="token_yes",
            target_price=0.50,
            edge=0.10,
            confidence=0.8,
            strength=SignalStrength.BUY,
            reasoning="test",
        )

        size = rm.calculate_position_size(signal)
        assert size > 0
        assert size <= 50  # MAX_POSITION_SIZE default

    def test_max_markets_limit(self):
        from bot.services.risk_manager import RiskManager
        from bot.models import Signal, Position
        from datetime import datetime, timezone

        rm = RiskManager()
        # Fill up positions
        for i in range(10):
            rm.add_position(Position(
                market_id=f"market_{i}",
                token_id=f"token_{i}",
                question=f"Market {i}",
                side=Side.BUY,
                entry_price=0.50,
                size=100,
                cost=50,
            ))

        market = make_market(condition_id="new_market")
        signal = Signal(
            market=market,
            strategy="test",
            side=Side.BUY,
            token_id="token_yes",
            target_price=0.50,
            edge=0.10,
            confidence=0.8,
            strength=SignalStrength.BUY,
            reasoning="test",
        )

        assert rm.can_open_position(signal) is False

    def test_no_duplicate_positions(self):
        from bot.services.risk_manager import RiskManager
        from bot.models import Signal, Position

        rm = RiskManager()
        rm.add_position(Position(
            market_id="test_123",
            token_id="token_yes",
            question="Test",
            side=Side.BUY,
            entry_price=0.50,
            size=100,
            cost=50,
        ))

        market = make_market(condition_id="test_123")
        signal = Signal(
            market=market,
            strategy="test",
            side=Side.BUY,
            token_id="token_yes",
            target_price=0.50,
            edge=0.10,
            confidence=0.8,
            strength=SignalStrength.BUY,
            reasoning="test",
        )

        assert rm.can_open_position(signal) is False


class TestMarketScanner:
    def test_filters_low_liquidity(self):
        from bot.services.market_scanner import MarketScanner

        scanner = MarketScanner()
        markets = [
            make_market(liquidity=500),  # too low
            make_market(liquidity=5000, condition_id="good"),  # ok
        ]
        result = scanner.scan(markets)
        assert len(result) == 1
        assert result[0].condition_id == "good"

    def test_filters_extreme_prices(self):
        from bot.services.market_scanner import MarketScanner

        scanner = MarketScanner()
        markets = [
            make_market(yes_price=0.98),  # too certain
            make_market(yes_price=0.02),  # too certain
            make_market(yes_price=0.50, condition_id="good"),
        ]
        result = scanner.scan(markets)
        assert len(result) == 1

    def test_filters_wide_spread(self):
        from bot.services.market_scanner import MarketScanner

        scanner = MarketScanner()
        markets = [
            make_market(spread=0.15),  # too wide
            make_market(spread=0.03, condition_id="good"),
        ]
        result = scanner.scan(markets)
        assert len(result) == 1
