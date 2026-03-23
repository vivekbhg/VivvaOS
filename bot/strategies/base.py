"""Base strategy interface."""
from abc import ABC, abstractmethod

from bot.models import Market, Signal


class BaseStrategy(ABC):
    """All strategies must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""

    @abstractmethod
    def analyze(self, market: Market, orderbook: dict) -> Signal | None:
        """Analyze a market and return a Signal if there's an opportunity, else None."""

    def _calculate_book_depth(self, orderbook: dict, side: str) -> float:
        """Calculate total depth on one side of the book."""
        entries = orderbook.get(side, [])
        if isinstance(entries, list):
            return sum(float(e.get("size", 0)) for e in entries)
        return 0.0

    def _weighted_avg_price(self, entries: list[dict], depth: float) -> float:
        """Calculate volume-weighted average price from orderbook entries."""
        if not entries or depth == 0:
            return 0.0
        total = sum(float(e.get("price", 0)) * float(e.get("size", 0)) for e in entries)
        return total / depth
