"""Social media monitoring agent - tracks Twitter/X, Reddit for market sentiment."""
import asyncio
import logging
import re
from datetime import datetime, timezone

import httpx

from bot.agents.base import BaseAgent, WebSignal, Sentiment

logger = logging.getLogger(__name__)

# Public Reddit JSON endpoints (no auth needed)
SUBREDDITS = [
    "polymarket",
    "prediction_market",
    "politics",
    "worldnews",
    "cryptocurrency",
    "economics",
    "geopolitics",
]

# Nitter instances for public Twitter scraping (fallback)
NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
]

# Polymarket-specific accounts to watch
TWITTER_ACCOUNTS = [
    "Polymarket",
    "Starkie",  # whale tracker
    "DustinMoskovitz",
    "elikikosk",
]


class SocialAgent(BaseAgent):
    """Monitors Reddit and Twitter for market-relevant social signals."""

    @property
    def name(self) -> str:
        return "social"

    @property
    def poll_interval_seconds(self) -> int:
        return 180  # Every 3 minutes

    async def poll(self, market_queries: list[str]) -> list[WebSignal]:
        """Poll Reddit and Twitter for signals."""
        signals = []

        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "VivvaOS/1.0 (Market Monitor)"},
        ) as client:
            # Run Reddit and Twitter monitoring concurrently
            reddit_task = self._poll_reddit(client, market_queries)
            twitter_task = self._poll_twitter(client, market_queries)

            reddit_signals, twitter_signals = await asyncio.gather(
                reddit_task, twitter_task, return_exceptions=True
            )

            if isinstance(reddit_signals, list):
                signals.extend(reddit_signals)
            else:
                logger.debug(f"[social] Reddit poll failed: {reddit_signals}")

            if isinstance(twitter_signals, list):
                signals.extend(twitter_signals)
            else:
                logger.debug(f"[social] Twitter poll failed: {twitter_signals}")

        return signals

    # ---- Reddit ----

    async def _poll_reddit(
        self, client: httpx.AsyncClient, market_queries: list[str]
    ) -> list[WebSignal]:
        """Poll Reddit for new posts mentioning market topics."""
        signals = []
        keywords = self._query_to_keywords(market_queries)

        tasks = [
            self._fetch_subreddit(client, sub, keywords, market_queries)
            for sub in SUBREDDITS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                signals.extend(result)

        return signals

    async def _fetch_subreddit(
        self,
        client: httpx.AsyncClient,
        subreddit: str,
        keywords: dict[str, str],
        queries: list[str],
    ) -> list[WebSignal]:
        """Fetch recent posts from a subreddit and match to markets."""
        signals = []
        try:
            url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=25"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            posts = data.get("data", {}).get("children", [])
            for post in posts:
                pdata = post.get("data", {})
                title = pdata.get("title", "")
                selftext = pdata.get("selftext", "")[:300]
                score = pdata.get("score", 0)
                num_comments = pdata.get("num_comments", 0)
                permalink = pdata.get("permalink", "")
                created = pdata.get("created_utc", 0)

                # Skip old posts (> 6 hours)
                if created and (datetime.now(timezone.utc).timestamp() - created) > 21600:
                    continue

                # Match to market queries
                combined = f"{title} {selftext}".lower()
                matched = self._match_text(combined, keywords)
                if not matched:
                    continue

                # Engagement = urgency proxy
                engagement = min(1.0, (score / 500) * 0.4 + (num_comments / 100) * 0.4 + 0.2)

                sentiment = self._analyze_sentiment(combined)
                relevance = self._calc_relevance(combined, matched)

                signals.append(WebSignal(
                    source=f"reddit/r/{subreddit}",
                    market_query=matched,
                    headline=title[:200],
                    url=f"https://reddit.com{permalink}",
                    sentiment=sentiment,
                    relevance=relevance,
                    urgency=engagement,
                ))

        except Exception as e:
            logger.debug(f"[social] Reddit r/{subreddit} failed: {e}")

        return signals

    # ---- Twitter/Nitter ----

    async def _poll_twitter(
        self, client: httpx.AsyncClient, market_queries: list[str]
    ) -> list[WebSignal]:
        """Poll Nitter (public Twitter proxy) for tweets from key accounts."""
        signals = []
        keywords = self._query_to_keywords(market_queries)

        for instance in NITTER_INSTANCES:
            try:
                for account in TWITTER_ACCOUNTS:
                    tweets = await self._fetch_nitter(client, instance, account)
                    for tweet in tweets:
                        matched = self._match_text(tweet["text"].lower(), keywords)
                        if matched:
                            sentiment = self._analyze_sentiment(tweet["text"].lower())
                            signals.append(WebSignal(
                                source=f"twitter/@{account}",
                                market_query=matched,
                                headline=tweet["text"][:200],
                                url=tweet.get("url", f"{instance}/{account}"),
                                sentiment=sentiment,
                                relevance=0.7,  # Key accounts = high relevance
                                urgency=0.6,
                            ))
                # If first instance works, don't try others
                if signals or True:
                    break
            except Exception as e:
                logger.debug(f"[social] Nitter {instance} failed: {e}")
                continue

        return signals

    async def _fetch_nitter(
        self, client: httpx.AsyncClient, instance: str, account: str
    ) -> list[dict]:
        """Scrape tweets from Nitter RSS feed."""
        tweets = []
        try:
            url = f"{instance}/{account}/rss"
            resp = await client.get(url)
            resp.raise_for_status()

            # Parse RSS
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:10]:
                title_el = item.find("title")
                link_el = item.find("link")
                if title_el is not None and title_el.text:
                    tweets.append({
                        "text": title_el.text.strip(),
                        "url": link_el.text.strip() if link_el is not None and link_el.text else "",
                    })
        except Exception:
            pass
        return tweets

    # ---- Shared helpers ----

    def _query_to_keywords(self, queries: list[str]) -> dict[str, str]:
        """Extract keywords from market queries."""
        stop_words = {
            "will", "the", "be", "to", "in", "of", "a", "an", "is", "it",
            "on", "for", "by", "at", "or", "and", "not", "with", "before",
            "after", "yes", "no", "this", "that", "than", "more", "less",
        }
        keywords = {}
        for query in queries:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            for word in words:
                if word not in stop_words:
                    keywords[word] = query
        return keywords

    def _match_text(self, text: str, keywords: dict[str, str]) -> str | None:
        """Match text against keywords, return matched query if >= 2 matches."""
        query_hits: dict[str, int] = {}
        for kw, query in keywords.items():
            if kw in text:
                query_hits[query] = query_hits.get(query, 0) + 1

        if not query_hits:
            return None

        best_query = max(query_hits, key=query_hits.get)
        if query_hits[best_query] >= 2:
            return best_query
        return None

    def _analyze_sentiment(self, text: str) -> Sentiment:
        """Simple keyword-based sentiment analysis."""
        bullish = [
            "bullish", "moon", "pump", "rally", "win", "winning", "up",
            "surge", "confirmed", "passed", "approved", "deal", "peace",
            "buy", "long", "undervalued", "cheap", "opportunity",
        ]
        bearish = [
            "bearish", "dump", "crash", "sell", "down", "plunge", "fail",
            "rejected", "war", "crisis", "overvalued", "expensive",
            "short", "scam", "fraud", "dead", "collapse",
        ]

        bull = sum(1 for w in bullish if w in text)
        bear = sum(1 for w in bearish if w in text)

        if bull > bear + 2:
            return Sentiment.VERY_BULLISH
        elif bull > bear:
            return Sentiment.BULLISH
        elif bear > bull + 2:
            return Sentiment.VERY_BEARISH
        elif bear > bull:
            return Sentiment.BEARISH
        return Sentiment.NEUTRAL

    def _calc_relevance(self, text: str, query: str) -> float:
        """Calculate relevance of text to a query."""
        query_words = set(query.lower().split())
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        return min(1.0, 0.3 + overlap * 0.12)
