"""Agent orchestrator - runs all web monitoring agents in parallel."""
import asyncio
import logging
from datetime import datetime, timezone

from bot.agents.base import BaseAgent, WebSignal, Sentiment
from bot.agents.news_agent import NewsAgent
from bot.agents.social_agent import SocialAgent
from bot.agents.odds_agent import OddsAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Manages all web monitoring agents. Runs them concurrently,
    collects signals, and provides a unified feed to the trading engine.
    """

    def __init__(self):
        self.agents: list[BaseAgent] = [
            NewsAgent(),
            SocialAgent(),
            OddsAgent(),
        ]
        self._all_signals: list[WebSignal] = []
        self._running = False
        self._background_task: asyncio.Task | None = None

    @property
    def signal_count(self) -> int:
        return len(self._all_signals)

    def start_background(self, market_queries: list[str], loop: asyncio.AbstractEventLoop = None):
        """Start agents running in the background on their own poll intervals."""
        self._running = True
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        self._background_task = loop.create_task(self._run_loop(market_queries))
        logger.info(f"Agent orchestrator started with {len(self.agents)} agents")

    def stop(self):
        """Stop all agents."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
        logger.info("Agent orchestrator stopped")

    async def poll_all(self, market_queries: list[str]) -> list[WebSignal]:
        """Run all agents once and return combined signals."""
        tasks = [agent.run_once(market_queries) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_signals = []
        for agent, result in zip(self.agents, results):
            if isinstance(result, list):
                new_signals.extend(result)
                logger.info(f"[{agent.name}] returned {len(result)} signals")
            elif isinstance(result, Exception):
                logger.error(f"[{agent.name}] failed: {result}")

        self._all_signals.extend(new_signals)
        self._prune_old_signals()
        return new_signals

    def poll_all_sync(self, market_queries: list[str]) -> list[WebSignal]:
        """Synchronous wrapper for poll_all."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.poll_all(market_queries))
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(self.poll_all(market_queries))
        except RuntimeError:
            return asyncio.run(self.poll_all(market_queries))

    async def _run_loop(self, market_queries: list[str]):
        """Background loop that polls agents at their own intervals."""
        last_poll: dict[str, float] = {}

        while self._running:
            now = datetime.now(timezone.utc).timestamp()

            for agent in self.agents:
                last = last_poll.get(agent.name, 0)
                if now - last >= agent.poll_interval_seconds:
                    try:
                        signals = await agent.run_once(market_queries)
                        self._all_signals.extend(signals)
                        last_poll[agent.name] = now
                    except Exception as e:
                        logger.error(f"[{agent.name}] background poll failed: {e}")

            self._prune_old_signals()
            await asyncio.sleep(10)  # Check every 10 seconds

    def get_signals_for_market(self, market_query: str, max_age_minutes: int = 30) -> list[WebSignal]:
        """Get web signals relevant to a specific market."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_minutes * 60)
        matching = []
        query_lower = market_query.lower()
        for sig in self._all_signals:
            if sig.timestamp.timestamp() < cutoff:
                continue
            if sig.market_query.lower() == query_lower:
                matching.append(sig)
        return sorted(matching, key=lambda s: s.impact_score, reverse=True)

    def get_top_signals(self, n: int = 20, max_age_minutes: int = 30) -> list[WebSignal]:
        """Get the top N highest-impact signals across all agents."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_minutes * 60)
        recent = [s for s in self._all_signals if s.timestamp.timestamp() > cutoff]
        return sorted(recent, key=lambda s: s.impact_score, reverse=True)[:n]

    def get_market_sentiment(self, market_query: str, max_age_minutes: int = 30) -> dict:
        """
        Aggregate sentiment for a market from all web signals.
        Returns sentiment summary dict.
        """
        signals = self.get_signals_for_market(market_query, max_age_minutes)

        if not signals:
            return {
                "sentiment": Sentiment.NEUTRAL,
                "confidence": 0.0,
                "signal_count": 0,
                "bullish": 0,
                "bearish": 0,
                "neutral": 0,
                "top_signal": None,
            }

        bullish = sum(1 for s in signals if s.is_bullish)
        bearish = sum(1 for s in signals if s.is_bearish)
        neutral = len(signals) - bullish - bearish

        # Weighted sentiment (impact-weighted)
        bull_weight = sum(s.impact_score for s in signals if s.is_bullish)
        bear_weight = sum(s.impact_score for s in signals if s.is_bearish)
        total_weight = bull_weight + bear_weight

        if total_weight == 0:
            agg_sentiment = Sentiment.NEUTRAL
            confidence = 0.0
        elif bull_weight > bear_weight * 1.5:
            agg_sentiment = Sentiment.VERY_BULLISH if bull_weight > bear_weight * 3 else Sentiment.BULLISH
            confidence = bull_weight / (total_weight + 1)
        elif bear_weight > bull_weight * 1.5:
            agg_sentiment = Sentiment.VERY_BEARISH if bear_weight > bull_weight * 3 else Sentiment.BEARISH
            confidence = bear_weight / (total_weight + 1)
        else:
            agg_sentiment = Sentiment.NEUTRAL
            confidence = 0.2

        top = max(signals, key=lambda s: s.impact_score)

        return {
            "sentiment": agg_sentiment,
            "confidence": min(1.0, confidence),
            "signal_count": len(signals),
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "top_signal": top,
        }

    def _prune_old_signals(self):
        """Remove signals older than 2 hours."""
        cutoff = datetime.now(timezone.utc).timestamp() - 7200
        self._all_signals = [s for s in self._all_signals if s.timestamp.timestamp() > cutoff]
        for agent in self.agents:
            agent.clear_old_signals(120)

    def get_agent_status(self) -> list[dict]:
        """Get status of each agent."""
        return [
            {
                "name": agent.name,
                "poll_interval": agent.poll_interval_seconds,
                "last_run": agent._last_run.isoformat() if agent._last_run else "never",
                "signal_count": len(agent.get_recent_signals()),
            }
            for agent in self.agents
        ]
