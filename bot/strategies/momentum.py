"""Momentum strategy - trade in the direction of volume-weighted price movement."""
import logging

from config.settings import settings
from bot.models import Market, Signal, Side, SignalStrength
from bot.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Detects momentum by analyzing orderbook pressure and volume.
    High volume + directional book pressure = momentum signal.

    Key indicators:
    - Bid wall detection (large bids near top = bullish)
    - Ask wall detection (large asks near top = bearish)
    - Volume relative to liquidity (high ratio = active market)
    """

    @property
    def name(self) -> str:
        return "momentum"

    def analyze(self, market: Market, orderbook: dict) -> Signal | None:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return None

        # Volume/liquidity ratio - high means active trading
        if market.liquidity == 0:
            return None
        activity_ratio = market.volume_24h / market.liquidity

        # Need meaningful activity
        if activity_ratio < 0.5:
            return None

        # Detect walls: large orders in the top 3 levels
        top_bids = sorted(bids, key=lambda x: float(x.get("price", 0)), reverse=True)[:3]
        top_asks = sorted(asks, key=lambda x: float(x.get("price", 0)))[:3]

        bid_wall = max((float(b.get("size", 0)) for b in top_bids), default=0)
        ask_wall = max((float(a.get("size", 0)) for a in top_asks), default=0)

        total_top = bid_wall + ask_wall
        if total_top == 0:
            return None

        # Wall ratio: positive = bid wall dominant, negative = ask wall dominant
        wall_ratio = (bid_wall - ask_wall) / total_top

        # Need clear directional signal
        if abs(wall_ratio) < 0.3:
            return None

        # Calculate edge from wall analysis + activity
        edge = abs(wall_ratio) * 0.15 * min(activity_ratio, 3.0) / 3.0

        if edge < settings.MIN_EDGE:
            return None

        if wall_ratio > 0:
            # Bullish momentum - buy YES
            side = Side.BUY
            token_id = market.token_id_yes
            target = min(0.95, market.yes_price + edge)
            reasoning = (
                f"Bullish momentum: bid wall ${bid_wall:.0f} vs ask wall ${ask_wall:.0f}. "
                f"Activity ratio {activity_ratio:.1f}x. Wall ratio {wall_ratio:+.2f}"
            )
        else:
            # Bearish momentum - buy NO
            side = Side.BUY
            token_id = market.token_id_no
            target = min(0.95, market.no_price + edge)
            reasoning = (
                f"Bearish momentum: ask wall ${ask_wall:.0f} vs bid wall ${bid_wall:.0f}. "
                f"Activity ratio {activity_ratio:.1f}x. Wall ratio {wall_ratio:+.2f}"
            )

        confidence = min(0.8, abs(wall_ratio) * 0.4 + min(activity_ratio, 3) * 0.15)

        strength = SignalStrength.STRONG_BUY if edge > 0.08 else SignalStrength.BUY

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
