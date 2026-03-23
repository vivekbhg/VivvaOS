"""Odds monitoring agent - tracks other prediction platforms for cross-market arbitrage."""
import asyncio
import logging
from datetime import datetime, timezone

import httpx

from bot.agents.base import BaseAgent, WebSignal, Sentiment

logger = logging.getLogger(__name__)

# Public prediction market / odds APIs
MANIFOLD_API = "https://api.manifold.markets/v0"
METACULUS_API = "https://www.metaculus.com/api2"


class OddsAgent(BaseAgent):
    """
    Monitors other prediction markets for cross-platform price discrepancies.

    If Polymarket has YES at $0.40 but Manifold has it at 60%, there's
    a potential edge. The crowd on the other platform may know something,
    or Polymarket may be slow to reprice.
    """

    @property
    def name(self) -> str:
        return "odds"

    @property
    def poll_interval_seconds(self) -> int:
        return 300  # Every 5 minutes

    async def poll(self, market_queries: list[str]) -> list[WebSignal]:
        """Search other platforms for matching markets and compare odds."""
        signals = []

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            # Search both platforms concurrently
            manifold_task = self._poll_manifold(client, market_queries)
            metaculus_task = self._poll_metaculus(client, market_queries)

            manifold_signals, metaculus_signals = await asyncio.gather(
                manifold_task, metaculus_task, return_exceptions=True
            )

            if isinstance(manifold_signals, list):
                signals.extend(manifold_signals)
            if isinstance(metaculus_signals, list):
                signals.extend(metaculus_signals)

        return signals

    async def _poll_manifold(
        self, client: httpx.AsyncClient, queries: list[str]
    ) -> list[WebSignal]:
        """Search Manifold Markets for matching predictions."""
        signals = []

        for query in queries[:10]:  # Cap to avoid rate limits
            try:
                # Search Manifold for similar markets
                search_terms = self._shorten_query(query)
                resp = await client.get(
                    f"{MANIFOLD_API}/search-markets",
                    params={"term": search_terms, "limit": 3},
                )
                if resp.status_code != 200:
                    continue

                markets = resp.json()
                for market in markets:
                    if market.get("outcomeType") != "BINARY":
                        continue
                    if market.get("isResolved", False):
                        continue

                    manifold_prob = market.get("probability", 0)
                    market_url = f"https://manifold.markets/{market.get('creatorUsername', '')}/{market.get('slug', '')}"
                    question = market.get("question", "")

                    # Check if this is actually about the same thing
                    similarity = self._text_similarity(query, question)
                    if similarity < 0.3:
                        continue

                    signals.append(WebSignal(
                        source="manifold",
                        market_query=query,
                        headline=f"Manifold: {question[:150]} @ {manifold_prob:.0%}",
                        url=market_url,
                        sentiment=Sentiment.NEUTRAL,
                        relevance=similarity,
                        urgency=0.4,
                    ))
                    # Store the probability for later comparison
                    signals[-1]._manifold_prob = manifold_prob

                # Small delay between searches
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.debug(f"[odds] Manifold search failed for '{query[:30]}': {e}")

        return signals

    async def _poll_metaculus(
        self, client: httpx.AsyncClient, queries: list[str]
    ) -> list[WebSignal]:
        """Search Metaculus for matching predictions."""
        signals = []

        for query in queries[:10]:
            try:
                search_terms = self._shorten_query(query)
                resp = await client.get(
                    f"{METACULUS_API}/questions/",
                    params={
                        "search": search_terms,
                        "limit": 3,
                        "type": "forecast",
                        "status": "open",
                    },
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                questions = data.get("results", [])

                for q in questions:
                    community_prediction = q.get("community_prediction", {})
                    if not community_prediction:
                        continue
                    full = community_prediction.get("full", {})
                    q2 = full.get("q2")  # median prediction
                    if q2 is None:
                        continue

                    question_text = q.get("title", "")
                    question_url = f"https://www.metaculus.com/questions/{q.get('id', '')}/"

                    similarity = self._text_similarity(query, question_text)
                    if similarity < 0.3:
                        continue

                    signals.append(WebSignal(
                        source="metaculus",
                        market_query=query,
                        headline=f"Metaculus: {question_text[:150]} @ {q2:.0%}",
                        url=question_url,
                        sentiment=Sentiment.NEUTRAL,
                        relevance=similarity,
                        urgency=0.3,
                    ))
                    signals[-1]._metaculus_prob = q2

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.debug(f"[odds] Metaculus search failed for '{query[:30]}': {e}")

        return signals

    def _shorten_query(self, query: str) -> str:
        """Shorten a market question to key search terms."""
        stop_words = {
            "will", "the", "be", "to", "in", "of", "a", "an", "is",
            "before", "after", "by", "on", "at", "or", "and",
        }
        words = query.split()
        important = [w for w in words if w.lower() not in stop_words]
        return " ".join(important[:6])

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-overlap similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        stop_words = {"will", "the", "be", "to", "in", "of", "a", "an", "is", "by", "on"}
        words1 -= stop_words
        words2 -= stop_words
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def compare_with_polymarket(
        self, web_signal: WebSignal, polymarket_price: float
    ) -> WebSignal | None:
        """
        Compare cross-platform odds with Polymarket price.
        Returns an enhanced signal if there's a meaningful discrepancy.
        """
        other_prob = getattr(web_signal, "_manifold_prob", None) or getattr(
            web_signal, "_metaculus_prob", None
        )
        if other_prob is None:
            return None

        discrepancy = other_prob - polymarket_price

        if abs(discrepancy) < 0.05:
            return None  # Within noise

        if discrepancy > 0:
            # Other platform thinks YES is more likely -> Polymarket YES is cheap
            web_signal.sentiment = (
                Sentiment.VERY_BULLISH if discrepancy > 0.15 else Sentiment.BULLISH
            )
            web_signal.headline = (
                f"Cross-market edge: {web_signal.source} @ {other_prob:.0%} "
                f"vs Polymarket @ {polymarket_price:.0%} (+{discrepancy:.0%})"
            )
        else:
            # Other platform thinks YES is less likely -> Polymarket YES is expensive
            web_signal.sentiment = (
                Sentiment.VERY_BEARISH if discrepancy < -0.15 else Sentiment.BEARISH
            )
            web_signal.headline = (
                f"Cross-market edge: {web_signal.source} @ {other_prob:.0%} "
                f"vs Polymarket @ {polymarket_price:.0%} ({discrepancy:+.0%})"
            )

        web_signal.relevance = min(1.0, abs(discrepancy) * 3)
        web_signal.urgency = 0.7  # Cross-market arb is fairly urgent
        return web_signal
