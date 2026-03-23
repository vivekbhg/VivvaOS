"""Risk management: position sizing, exposure limits, stop-loss/take-profit."""
import logging

from config.settings import settings
from bot.models import Signal, Position, Side

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk across all positions."""

    def __init__(self):
        self.positions: list[Position] = []
        self.trade_history: list[dict] = []

    @property
    def total_exposure(self) -> float:
        return sum(p.cost for p in self.positions)

    @property
    def total_pnl(self) -> float:
        return sum(p.pnl for p in self.positions)

    def can_open_position(self, signal: Signal) -> bool:
        """Check if we can open a new position given current risk limits."""
        if len(self.positions) >= settings.MAX_MARKETS:
            logger.debug(f"Max markets reached ({settings.MAX_MARKETS})")
            return False

        if self.total_exposure >= settings.MAX_TOTAL_EXPOSURE:
            logger.debug(f"Max exposure reached (${self.total_exposure:.2f})")
            return False

        # Don't double up on same market
        for p in self.positions:
            if p.market_id == signal.market.condition_id:
                logger.debug(f"Already have position in {signal.market.question[:40]}")
                return False

        return True

    def calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size in USD based on signal quality and risk limits."""
        # Base size from config
        max_size = settings.MAX_POSITION_SIZE

        # Scale by remaining exposure room
        remaining = settings.MAX_TOTAL_EXPOSURE - self.total_exposure
        max_size = min(max_size, remaining)

        # Scale by confidence (higher confidence = larger position)
        size = max_size * signal.confidence

        # Scale by edge (higher edge = larger position)
        edge_multiplier = min(2.0, signal.edge / settings.MIN_EDGE)
        size *= min(1.0, edge_multiplier * 0.5)

        # Minimum trade size of $5
        if size < 5.0:
            return 0.0

        # Round to 2 decimals
        return round(size, 2)

    def add_position(self, position: Position):
        """Track a new position."""
        self.positions.append(position)
        logger.info(
            f"Opened: {position.side.value} {position.question[:40]}... "
            f"@ ${position.entry_price:.3f} x{position.size:.1f} (${position.cost:.2f})"
        )

    def remove_position(self, market_id: str) -> Position | None:
        """Remove and return a position by market ID."""
        for i, p in enumerate(self.positions):
            if p.market_id == market_id:
                return self.positions.pop(i)
        return None

    def check_exits(self) -> list[Position]:
        """Check all positions for stop-loss or take-profit triggers."""
        exits = []
        for pos in self.positions:
            if pos.should_stop_loss:
                logger.warning(
                    f"STOP LOSS: {pos.question[:40]}... PnL: {pos.pnl_pct:+.1%} (${pos.pnl:+.2f})"
                )
                exits.append(pos)
            elif pos.should_take_profit:
                logger.info(
                    f"TAKE PROFIT: {pos.question[:40]}... PnL: {pos.pnl_pct:+.1%} (${pos.pnl:+.2f})"
                )
                exits.append(pos)
        return exits

    def get_portfolio_summary(self) -> dict:
        """Generate portfolio summary stats."""
        if not self.positions:
            return {
                "positions": 0,
                "total_exposure": 0,
                "total_pnl": 0,
                "pnl_pct": 0,
                "best": None,
                "worst": None,
            }

        best = max(self.positions, key=lambda p: p.pnl_pct)
        worst = min(self.positions, key=lambda p: p.pnl_pct)

        return {
            "positions": len(self.positions),
            "total_exposure": self.total_exposure,
            "total_pnl": self.total_pnl,
            "pnl_pct": self.total_pnl / max(self.total_exposure, 1),
            "best": {"question": best.question[:50], "pnl_pct": best.pnl_pct},
            "worst": {"question": worst.question[:50], "pnl_pct": worst.pnl_pct},
        }
