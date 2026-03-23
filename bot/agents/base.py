"""Base class for all web monitoring agents."""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


@dataclass
class WebSignal:
    """A signal derived from web monitoring."""
    source: str              # which agent produced this
    market_query: str        # search term / market question it relates to
    headline: str            # what was found
    url: str                 # source URL
    sentiment: Sentiment
    relevance: float         # 0-1, how relevant to the market
    urgency: float           # 0-1, how time-sensitive (breaking news = high)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def impact_score(self) -> float:
        """Combined score: relevance * urgency * sentiment magnitude."""
        sentiment_map = {
            Sentiment.VERY_BULLISH: 1.0,
            Sentiment.BULLISH: 0.6,
            Sentiment.NEUTRAL: 0.0,
            Sentiment.BEARISH: 0.6,
            Sentiment.VERY_BEARISH: 1.0,
        }
        return self.relevance * self.urgency * sentiment_map.get(self.sentiment, 0)

    @property
    def is_bullish(self) -> bool:
        return self.sentiment in (Sentiment.BULLISH, Sentiment.VERY_BULLISH)

    @property
    def is_bearish(self) -> bool:
        return self.sentiment in (Sentiment.BEARISH, Sentiment.VERY_BEARISH)


class BaseAgent(ABC):
    """Base web monitoring agent. Each agent watches a specific source."""

    def __init__(self):
        self._running = False
        self._signals: list[WebSignal] = []
        self._last_run: datetime | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""

    @property
    @abstractmethod
    def poll_interval_seconds(self) -> int:
        """How often to poll (in seconds)."""

    @abstractmethod
    async def poll(self, market_queries: list[str]) -> list[WebSignal]:
        """
        Poll the source for signals related to the given market queries.
        market_queries = list of keywords/questions from active markets.
        Returns new signals found.
        """

    async def run_once(self, market_queries: list[str]) -> list[WebSignal]:
        """Run a single poll cycle."""
        try:
            signals = await self.poll(market_queries)
            self._signals.extend(signals)
            self._last_run = datetime.now(timezone.utc)
            if signals:
                logger.info(f"[{self.name}] Found {len(signals)} signals")
            return signals
        except Exception as e:
            logger.error(f"[{self.name}] Poll failed: {e}")
            return []

    def get_recent_signals(self, max_age_minutes: int = 30) -> list[WebSignal]:
        """Get signals from the last N minutes."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_minutes * 60)
        return [s for s in self._signals if s.timestamp.timestamp() > cutoff]

    def clear_old_signals(self, max_age_minutes: int = 120):
        """Prune signals older than N minutes."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_minutes * 60)
        self._signals = [s for s in self._signals if s.timestamp.timestamp() > cutoff]
