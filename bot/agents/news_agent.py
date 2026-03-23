"""News monitoring agent - scrapes RSS feeds and news APIs for market-moving events."""
import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from bot.agents.base import BaseAgent, WebSignal, Sentiment

logger = logging.getLogger(__name__)

# Major RSS feeds covering politics, economics, world events
RSS_FEEDS = {
    "reuters_world": "https://feeds.reuters.com/Reuters/worldNews",
    "reuters_politics": "https://feeds.reuters.com/Reuters/PoliticsNews",
    "reuters_business": "https://feeds.reuters.com/Reuters/businessNews",
    "ap_topnews": "https://rsshub.app/apnews/topics/apf-topnews",
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "nyt_world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
}

# Keywords that signal urgency (breaking news, elections, disasters, etc.)
URGENCY_KEYWORDS = [
    "breaking", "just in", "urgent", "developing", "alert",
    "confirmed", "official", "dead", "killed", "attack",
    "crash", "surge", "plunge", "resign", "impeach",
    "elected", "wins", "loses", "defeated", "victory",
    "ceasefire", "invasion", "war", "sanctions", "ban",
]

# Sentiment keyword buckets
BULLISH_KEYWORDS = [
    "wins", "victory", "approved", "passes", "success", "surge",
    "growth", "deal", "agreement", "peace", "ceasefire", "recovery",
    "breakthrough", "elected", "confirmed", "supports", "boost",
]

BEARISH_KEYWORDS = [
    "loses", "defeat", "rejected", "fails", "crash", "collapse",
    "crisis", "war", "attack", "killed", "sanctions", "ban",
    "resign", "impeach", "scandal", "fraud", "recession", "decline",
]


class NewsAgent(BaseAgent):
    """Monitors RSS news feeds for market-relevant events."""

    @property
    def name(self) -> str:
        return "news"

    @property
    def poll_interval_seconds(self) -> int:
        return 120  # Every 2 minutes

    async def poll(self, market_queries: list[str]) -> list[WebSignal]:
        """Fetch RSS feeds and match headlines to market queries."""
        signals = []

        # Build keyword index from market queries
        keywords = self._extract_keywords(market_queries)
        if not keywords:
            return signals

        # Fetch all feeds concurrently
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            tasks = []
            for feed_name, feed_url in RSS_FEEDS.items():
                tasks.append(self._fetch_feed(client, feed_name, feed_url))
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process each feed's articles
        for feed_name, articles in zip(RSS_FEEDS.keys(), results):
            if isinstance(articles, Exception):
                logger.debug(f"[news] Feed {feed_name} failed: {articles}")
                continue
            for article in articles:
                matched_query = self._match_article(article, keywords, market_queries)
                if matched_query:
                    signal = self._article_to_signal(article, matched_query)
                    if signal.impact_score > 0.1:
                        signals.append(signal)

        # Deduplicate by headline similarity
        signals = self._deduplicate(signals)

        return signals

    async def _fetch_feed(self, client: httpx.AsyncClient, name: str, url: str) -> list[dict]:
        """Fetch and parse an RSS feed."""
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return self._parse_rss(resp.text, name)
        except Exception as e:
            logger.debug(f"[news] Failed to fetch {name}: {e}")
            return []

    def _parse_rss(self, xml_text: str, source: str) -> list[dict]:
        """Parse RSS XML into article dicts."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
            # Handle both RSS 2.0 and Atom feeds
            items = root.findall(".//item") or root.findall(
                ".//{http://www.w3.org/2005/Atom}entry"
            )
            for item in items[:20]:  # Cap at 20 per feed
                title = self._get_text(item, "title") or self._get_text(
                    item, "{http://www.w3.org/2005/Atom}title"
                )
                link = self._get_text(item, "link") or ""
                if not link:
                    link_el = item.find("{http://www.w3.org/2005/Atom}link")
                    link = link_el.get("href", "") if link_el is not None else ""
                desc = self._get_text(item, "description") or self._get_text(
                    item, "{http://www.w3.org/2005/Atom}summary"
                ) or ""
                pub_date = self._get_text(item, "pubDate") or self._get_text(
                    item, "{http://www.w3.org/2005/Atom}updated"
                ) or ""

                if title:
                    articles.append({
                        "title": title.strip(),
                        "link": link.strip(),
                        "description": desc.strip()[:500],
                        "pub_date": pub_date,
                        "source": source,
                    })
        except ET.ParseError:
            pass
        return articles

    def _get_text(self, element, tag: str) -> str | None:
        el = element.find(tag)
        return el.text if el is not None and el.text else None

    def _extract_keywords(self, market_queries: list[str]) -> dict[str, str]:
        """Extract searchable keywords from market questions.
        Returns {keyword: original_query}."""
        stop_words = {
            "will", "the", "be", "to", "in", "of", "a", "an", "is", "it",
            "on", "for", "by", "at", "or", "and", "not", "with", "this",
            "that", "from", "has", "have", "was", "were", "are", "been",
            "before", "after", "during", "than", "more", "less", "yes", "no",
        }
        keywords = {}
        for query in market_queries:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            for word in words:
                if word not in stop_words:
                    keywords[word] = query
            # Also extract named entities (capitalized multi-word phrases)
            entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
            for entity in entities:
                keywords[entity.lower()] = query
        return keywords

    def _match_article(self, article: dict, keywords: dict, queries: list[str]) -> str | None:
        """Check if an article matches any market query. Returns matched query or None."""
        text = f"{article['title']} {article['description']}".lower()
        best_match = None
        best_count = 0

        # Check keyword matches
        for keyword, query in keywords.items():
            if keyword in text:
                # Count how many keywords from this query match
                query_keywords = [k for k, q in keywords.items() if q == query]
                match_count = sum(1 for k in query_keywords if k in text)
                if match_count > best_count:
                    best_count = match_count
                    best_match = query

        # Require at least 2 keyword matches to reduce false positives
        if best_count >= 2:
            return best_match
        return None

    def _article_to_signal(self, article: dict, matched_query: str) -> WebSignal:
        """Convert a matched article into a WebSignal."""
        title_lower = article["title"].lower()
        desc_lower = article["description"].lower()
        combined = f"{title_lower} {desc_lower}"

        # Determine urgency
        urgency = 0.3  # base urgency
        for kw in URGENCY_KEYWORDS:
            if kw in combined:
                urgency = min(1.0, urgency + 0.2)

        # Determine sentiment
        bull_score = sum(1 for kw in BULLISH_KEYWORDS if kw in combined)
        bear_score = sum(1 for kw in BEARISH_KEYWORDS if kw in combined)

        if bull_score > bear_score + 1:
            sentiment = Sentiment.VERY_BULLISH if bull_score > 3 else Sentiment.BULLISH
        elif bear_score > bull_score + 1:
            sentiment = Sentiment.VERY_BEARISH if bear_score > 3 else Sentiment.BEARISH
        else:
            sentiment = Sentiment.NEUTRAL

        # Relevance based on title vs description match
        relevance = 0.5
        query_words = set(matched_query.lower().split())
        title_words = set(title_lower.split())
        overlap = len(query_words & title_words)
        relevance = min(1.0, 0.3 + overlap * 0.15)

        return WebSignal(
            source=f"news/{article['source']}",
            market_query=matched_query,
            headline=article["title"][:200],
            url=article["link"],
            sentiment=sentiment,
            relevance=relevance,
            urgency=urgency,
        )

    def _deduplicate(self, signals: list[WebSignal]) -> list[WebSignal]:
        """Remove near-duplicate signals by headline similarity."""
        seen = set()
        unique = []
        for sig in signals:
            # Simple dedup: first 8 words of headline
            key = " ".join(sig.headline.lower().split()[:8])
            if key not in seen:
                seen.add(key)
                unique.append(sig)
        return unique
