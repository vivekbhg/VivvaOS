"""Value strategy - find markets where price deviates from fair value signals."""
import logging

from config.settings import settings
from bot.models import Market, Signal, Side, SignalStrength
from bot.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ValueStrategy(BaseStrategy):
    """
    Identifies mispriced markets by comparing the current price to
    signals of fair value:
    - Bid/ask imbalance suggests directional pressure
    - Price vs midpoint divergence
    - Volume-weighted price vs displayed price
    """

    @property
    def name(self) -> str:
        return "value"

    def analyze(self, market: Market, orderbook: dict) -> Signal | None:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return None

        bid_depth = self._calculate_book_depth(orderbook, "bids")
        ask_depth = self._calculate_book_depth(orderbook, "asks")
        total_depth = bid_depth + ask_depth

        if total_depth == 0:
            return None

        # 1. Order book imbalance: if bids >> asks, price should go up
        imbalance = (bid_depth - ask_depth) / total_depth  # -1 to +1

        # 2. VWAP vs current price
        vwap_bid = self._weighted_avg_price(bids, bid_depth)
        vwap_ask = self._weighted_avg_price(asks, ask_depth)
        fair_value = (vwap_bid + vwap_ask) / 2 if vwap_bid and vwap_ask else market.midpoint

        # 3. Calculate edge
        current_price = market.yes_price
        edge = abs(fair_value - current_price)

        if edge < settings.MIN_EDGE:
            return None

        # Determine direction
        if fair_value > current_price and imbalance > 0.1:
            # Market is underpriced -> buy YES
            side = Side.BUY
            token_id = market.token_id_yes
            target = fair_value
            reasoning = (
                f"Underpriced: YES at ${current_price:.3f} vs fair value ${fair_value:.3f}. "
                f"Book imbalance {imbalance:+.2f} (bid-heavy). Edge: {edge:.1%}"
            )
        elif fair_value < current_price and imbalance < -0.1:
            # Market is overpriced -> buy NO (or sell YES)
            side = Side.BUY
            token_id = market.token_id_no
            target = 1 - fair_value
            edge = abs((1 - fair_value) - market.no_price)
            if edge < settings.MIN_EDGE:
                return None
            reasoning = (
                f"Overpriced: YES at ${current_price:.3f} vs fair value ${fair_value:.3f}. "
                f"Book imbalance {imbalance:+.2f} (ask-heavy). Edge: {edge:.1%}"
            )
        else:
            return None

        # Confidence based on depth and imbalance magnitude
        confidence = min(0.9, abs(imbalance) * 0.5 + (total_depth / 10000) * 0.3 + edge * 2)
        strength = SignalStrength.STRONG_BUY if edge > 0.10 else SignalStrength.BUY

        return Signal(
            market=market,
            strategy=self.name,
            side=side,
            token_id=token_id,
            target_price=target,
            edge=edge,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning,
        )
