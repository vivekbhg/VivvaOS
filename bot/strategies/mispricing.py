"""Mispricing strategy - exploit markets where YES + NO prices don't sum to ~1.0."""
import logging

from config.settings import settings
from bot.models import Market, Signal, Side, SignalStrength
from bot.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MispricingStrategy(BaseStrategy):
    """
    Exploits binary market mispricing:
    - In a perfect market, P(YES) + P(NO) = 1.0
    - When this sum deviates, there's a risk-free(ish) opportunity
    - Also detects when one side is clearly cheaper than it should be

    Types of mispricing:
    1. Overround: YES + NO > 1.0 (both sides overpriced - sell opportunity)
    2. Underround: YES + NO < 1.0 (both sides underpriced - buy opportunity)
    3. Skew: One side is much cheaper relative to the other
    """

    @property
    def name(self) -> str:
        return "mispricing"

    def analyze(self, market: Market, orderbook: dict) -> Signal | None:
        yes_price = market.yes_price
        no_price = market.no_price

        # Total probability (should be ~1.0 in efficient market)
        total = yes_price + no_price
        deviation = total - 1.0

        # Strategy 1: Underround (total < 1.0) - both sides are cheap
        if deviation < -settings.MIN_EDGE:
            edge = abs(deviation)
            # Buy the cheaper side (more room to grow)
            if yes_price < no_price:
                side = Side.BUY
                token_id = market.token_id_yes
                target = yes_price + edge / 2
                reasoning = (
                    f"Underround: YES(${yes_price:.3f}) + NO(${no_price:.3f}) = "
                    f"${total:.3f} < $1.00. Gap: ${edge:.3f}. Buying cheaper YES side."
                )
            else:
                side = Side.BUY
                token_id = market.token_id_no
                target = no_price + edge / 2
                reasoning = (
                    f"Underround: YES(${yes_price:.3f}) + NO(${no_price:.3f}) = "
                    f"${total:.3f} < $1.00. Gap: ${edge:.3f}. Buying cheaper NO side."
                )

            confidence = min(0.85, edge * 5)
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

        # Strategy 2: Skew detection
        # If YES = 0.30 and NO = 0.65, the NO side implies YES should be 0.35
        implied_yes = 1.0 - no_price
        implied_no = 1.0 - yes_price

        yes_skew = implied_yes - yes_price  # positive = YES is cheap
        no_skew = implied_no - no_price     # positive = NO is cheap

        max_skew = max(abs(yes_skew), abs(no_skew))
        if max_skew < settings.MIN_EDGE:
            return None

        if yes_skew > no_skew and yes_skew > settings.MIN_EDGE:
            # YES is cheaper than implied - buy YES
            edge = yes_skew
            return Signal(
                market=market,
                strategy=self.name,
                side=Side.BUY,
                token_id=market.token_id_yes,
                target_price=implied_yes,
                edge=edge,
                confidence=min(0.75, edge * 4),
                strength=SignalStrength.BUY,
                reasoning=(
                    f"YES skew: priced ${yes_price:.3f} but NO implies fair value "
                    f"${implied_yes:.3f}. Skew: ${yes_skew:.3f}"
                ),
            )
        elif no_skew > settings.MIN_EDGE:
            # NO is cheaper than implied - buy NO
            edge = no_skew
            return Signal(
                market=market,
                strategy=self.name,
                side=Side.BUY,
                token_id=market.token_id_no,
                target_price=implied_no,
                edge=edge,
                confidence=min(0.75, edge * 4),
                strength=SignalStrength.BUY,
                reasoning=(
                    f"NO skew: priced ${no_price:.3f} but YES implies fair value "
                    f"${implied_no:.3f}. Skew: ${no_skew:.3f}"
                ),
            )

        return None
