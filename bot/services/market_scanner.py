"""Scans Polymarket for tradeable opportunities."""
import logging
from datetime import datetime, timezone

from config.settings import settings
from bot.models import Market

logger = logging.getLogger(__name__)


class MarketScanner:
    """Filters markets for trading opportunities based on configurable criteria."""

    def __init__(self, min_liquidity: float = None, max_spread: float = None):
        self.min_liquidity = min_liquidity or settings.MIN_LIQUIDITY
        self.max_spread = max_spread or settings.MAX_SPREAD

    def scan(self, markets: list[Market]) -> list[Market]:
        """Filter markets to those worth analyzing."""
        candidates = []
        for market in markets:
            if not self._passes_filters(market):
                continue
            candidates.append(market)

        # Sort by volume (most active first)
        candidates.sort(key=lambda m: m.volume_24h, reverse=True)
        logger.info(f"Scanner: {len(candidates)} candidates from {len(markets)} markets")
        return candidates

    def _passes_filters(self, market: Market) -> bool:
        """Check if a market passes all trading filters."""
        # Must have valid token IDs
        if not market.token_id_yes or not market.token_id_no:
            return False

        # Minimum liquidity
        if market.liquidity < self.min_liquidity:
            return False

        # Maximum spread (too wide = too expensive to trade)
        if market.spread > self.max_spread:
            return False

        # Skip markets with extreme prices (already resolved or near-certain)
        if market.yes_price > 0.95 or market.yes_price < 0.05:
            return False

        # Skip markets with no volume
        if market.volume_24h < 100:
            return False

        # Skip expired markets
        if market.end_date:
            try:
                end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                if end < datetime.now(timezone.utc):
                    return False
            except (ValueError, TypeError):
                pass

        return True
