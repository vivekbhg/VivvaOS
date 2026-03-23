"""Tests for web monitoring agents and orchestrator."""
import pytest
from datetime import datetime, timezone

from bot.agents.base import WebSignal, Sentiment, BaseAgent
from bot.agents.news_agent import NewsAgent
from bot.agents.social_agent import SocialAgent
from bot.agents.odds_agent import OddsAgent
from bot.agents.orchestrator import AgentOrchestrator


class TestWebSignal:
    def test_impact_score_high(self):
        sig = WebSignal(
            source="test",
            market_query="Will X happen?",
            headline="Breaking: X confirmed",
            url="https://example.com",
            sentiment=Sentiment.VERY_BULLISH,
            relevance=0.9,
            urgency=0.8,
        )
        assert sig.impact_score > 0.5

    def test_impact_score_neutral(self):
        sig = WebSignal(
            source="test",
            market_query="test",
            headline="test",
            url="",
            sentiment=Sentiment.NEUTRAL,
            relevance=0.9,
            urgency=0.9,
        )
        assert sig.impact_score == 0.0

    def test_bullish_bearish(self):
        bull = WebSignal(
            source="test", market_query="test", headline="x", url="",
            sentiment=Sentiment.BULLISH, relevance=0.5, urgency=0.5,
        )
        bear = WebSignal(
            source="test", market_query="test", headline="x", url="",
            sentiment=Sentiment.VERY_BEARISH, relevance=0.5, urgency=0.5,
        )
        assert bull.is_bullish
        assert not bull.is_bearish
        assert bear.is_bearish
        assert not bear.is_bullish


class TestNewsAgent:
    def setup_method(self):
        self.agent = NewsAgent()

    def test_extract_keywords(self):
        queries = ["Will Donald Trump win the 2024 election?"]
        keywords = self.agent._extract_keywords(queries)
        assert "trump" in keywords or "donald" in keywords
        assert "election" in keywords
        # Stop words excluded
        assert "will" not in keywords
        assert "the" not in keywords

    def test_article_matching(self):
        keywords = {
            "trump": "Will Trump win?",
            "election": "Will Trump win?",
            "president": "Will Trump win?",
        }
        article = {
            "title": "Trump leads in election polls",
            "description": "The president is ahead in swing states",
            "source": "test",
            "link": "",
        }
        result = self.agent._match_article(article, keywords, ["Will Trump win?"])
        assert result == "Will Trump win?"

    def test_no_match_irrelevant(self):
        keywords = {"bitcoin": "Will Bitcoin hit 100k?"}
        article = {
            "title": "New recipe for pasta",
            "description": "A delicious Italian dish",
            "source": "test",
            "link": "",
        }
        result = self.agent._match_article(article, keywords, ["Will Bitcoin hit 100k?"])
        assert result is None

    def test_deduplication(self):
        signals = [
            WebSignal(
                source="a", market_query="test",
                headline="Breaking news the president confirmed the deal today sources say",
                url="", sentiment=Sentiment.BULLISH, relevance=0.5, urgency=0.5,
            ),
            WebSignal(
                source="b", market_query="test",
                headline="Breaking news the president confirmed the deal today per officials",
                url="", sentiment=Sentiment.BULLISH, relevance=0.5, urgency=0.5,
            ),
        ]
        result = self.agent._deduplicate(signals)
        assert len(result) == 1

    def test_parse_rss(self):
        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0">
        <channel>
            <item>
                <title>Test Headline</title>
                <link>https://example.com/1</link>
                <description>Test description here</description>
            </item>
        </channel>
        </rss>"""
        articles = self.agent._parse_rss(rss_xml, "test")
        assert len(articles) == 1
        assert articles[0]["title"] == "Test Headline"
        assert articles[0]["link"] == "https://example.com/1"


class TestSocialAgent:
    def setup_method(self):
        self.agent = SocialAgent()

    def test_sentiment_analysis_bullish(self):
        text = "this is so bullish, moon pump rally incoming"
        result = self.agent._analyze_sentiment(text)
        assert result in (Sentiment.BULLISH, Sentiment.VERY_BULLISH)

    def test_sentiment_analysis_bearish(self):
        text = "crash dump sell everything dead collapse"
        result = self.agent._analyze_sentiment(text)
        assert result in (Sentiment.BEARISH, Sentiment.VERY_BEARISH)

    def test_sentiment_analysis_neutral(self):
        text = "the weather is nice today"
        result = self.agent._analyze_sentiment(text)
        assert result == Sentiment.NEUTRAL

    def test_keyword_matching(self):
        keywords = {
            "bitcoin": "Will Bitcoin hit 100k?",
            "crypto": "Will Bitcoin hit 100k?",
        }
        text = "bitcoin and crypto markets are moving"
        result = self.agent._match_text(text, keywords)
        assert result == "Will Bitcoin hit 100k?"

    def test_no_match_single_keyword(self):
        keywords = {
            "bitcoin": "Will Bitcoin hit 100k?",
            "crypto": "Will Bitcoin hit 100k?",
        }
        text = "the weather is nice today bitcoin"
        # Only 1 keyword match, needs >= 2
        result = self.agent._match_text(text, keywords)
        assert result is None


class TestOddsAgent:
    def setup_method(self):
        self.agent = OddsAgent()

    def test_text_similarity(self):
        sim = self.agent._text_similarity(
            "Will Trump win the 2024 election?",
            "Trump wins the 2024 presidential election",
        )
        assert sim > 0.25

    def test_text_similarity_unrelated(self):
        sim = self.agent._text_similarity(
            "Will Bitcoin hit 100k?",
            "New pasta recipe from Italy",
        )
        assert sim < 0.1

    def test_shorten_query(self):
        result = self.agent._shorten_query("Will the United States pass the infrastructure bill by December?")
        assert "United" in result or "States" in result or "infrastructure" in result
        assert "will" not in result.lower().split() or len(result.split()) <= 6

    def test_compare_with_polymarket_big_gap(self):
        sig = WebSignal(
            source="manifold", market_query="test", headline="Manifold: test @ 70%",
            url="", sentiment=Sentiment.NEUTRAL, relevance=0.5, urgency=0.4,
        )
        sig._manifold_prob = 0.70

        result = self.agent.compare_with_polymarket(sig, 0.45)
        assert result is not None
        assert result.is_bullish  # Other platform thinks YES is more likely

    def test_compare_with_polymarket_no_gap(self):
        sig = WebSignal(
            source="manifold", market_query="test", headline="test",
            url="", sentiment=Sentiment.NEUTRAL, relevance=0.5, urgency=0.4,
        )
        sig._manifold_prob = 0.50

        result = self.agent.compare_with_polymarket(sig, 0.48)
        assert result is None  # Within noise


class TestOrchestrator:
    def test_init(self):
        orch = AgentOrchestrator()
        assert len(orch.agents) == 3
        names = [a.name for a in orch.agents]
        assert "news" in names
        assert "social" in names
        assert "odds" in names

    def test_get_market_sentiment_empty(self):
        orch = AgentOrchestrator()
        result = orch.get_market_sentiment("Some market question")
        assert result["sentiment"] == Sentiment.NEUTRAL
        assert result["signal_count"] == 0

    def test_get_market_sentiment_with_signals(self):
        orch = AgentOrchestrator()
        # Inject some signals
        orch._all_signals = [
            WebSignal(
                source="test", market_query="Will X happen?",
                headline="X confirmed", url="",
                sentiment=Sentiment.VERY_BULLISH, relevance=0.9, urgency=0.8,
            ),
            WebSignal(
                source="test2", market_query="Will X happen?",
                headline="X looking likely", url="",
                sentiment=Sentiment.BULLISH, relevance=0.7, urgency=0.5,
            ),
        ]
        result = orch.get_market_sentiment("Will X happen?")
        assert result["signal_count"] == 2
        assert result["bullish"] == 2
        assert result["sentiment"] in (Sentiment.BULLISH, Sentiment.VERY_BULLISH)

    def test_get_top_signals(self):
        orch = AgentOrchestrator()
        orch._all_signals = [
            WebSignal(
                source="a", market_query="q1", headline="low impact", url="",
                sentiment=Sentiment.NEUTRAL, relevance=0.1, urgency=0.1,
            ),
            WebSignal(
                source="b", market_query="q2", headline="high impact", url="",
                sentiment=Sentiment.VERY_BULLISH, relevance=0.9, urgency=0.9,
            ),
        ]
        top = orch.get_top_signals(n=1)
        assert len(top) == 1
        assert top[0].headline == "high impact"

    def test_agent_status(self):
        orch = AgentOrchestrator()
        status = orch.get_agent_status()
        assert len(status) == 3
        for s in status:
            assert "name" in s
            assert "poll_interval" in s
            assert "last_run" in s
