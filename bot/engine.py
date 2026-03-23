"""Core trading engine - orchestrates scanning, strategy, execution, and risk."""
import asyncio
import logging
import time
from datetime import datetime, timezone

from config.settings import settings
from bot.models import Market, Signal, Position, Side
from bot.services.polymarket_client import PolymarketClient
from bot.services.market_scanner import MarketScanner
from bot.services.risk_manager import RiskManager
from bot.services.notifier import Notifier
from bot.strategies.value import ValueStrategy
from bot.strategies.momentum import MomentumStrategy
from bot.strategies.mispricing import MispricingStrategy
from bot.agents.orchestrator import AgentOrchestrator
from bot.agents.base import Sentiment

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine that runs the bot loop."""

    def __init__(self, enable_agents: bool = True):
        self.client = PolymarketClient()
        self.scanner = MarketScanner()
        self.risk_manager = RiskManager()
        self.notifier = Notifier()
        self.agents = AgentOrchestrator() if enable_agents else None
        self.strategies = [
            (ValueStrategy(), settings.STRATEGY_VALUE),
            (MomentumStrategy(), settings.STRATEGY_MOMENTUM),
            (MispricingStrategy(), settings.STRATEGY_MISPRICING),
        ]
        self.cycle_count = 0
        self.total_trades = 0
        self.start_time = None
        self._running = False

    def start(self):
        """Initialize and authenticate."""
        logger.info("=" * 60)
        logger.info("VivvaOS Polymarket Bot Starting")
        logger.info(f"Mode: {'DRY RUN' if settings.DRY_RUN else 'LIVE TRADING'}")
        logger.info(f"Max position: ${settings.MAX_POSITION_SIZE}")
        logger.info(f"Max exposure: ${settings.MAX_TOTAL_EXPOSURE}")
        logger.info(f"Min edge: {settings.MIN_EDGE:.0%}")
        logger.info(f"Stop loss: {settings.STOP_LOSS:.0%} | Take profit: {settings.TAKE_PROFIT:.0%}")
        logger.info(f"Web agents: {'ENABLED' if self.agents else 'DISABLED'}")
        logger.info("=" * 60)

        self.client.authenticate()
        self.start_time = datetime.now(timezone.utc)
        self._running = True

    def stop(self):
        """Stop the engine."""
        self._running = False
        if self.agents:
            self.agents.stop()
        logger.info("Engine stopped")

    def run_cycle(self) -> dict:
        """Run one complete trading cycle. Returns cycle summary."""
        self.cycle_count += 1
        cycle_start = time.monotonic()

        logger.info(f"\n--- Cycle {self.cycle_count} ---")

        # 1. Fetch markets
        markets = self.client.get_markets()
        if not markets:
            logger.warning("No markets fetched, skipping cycle")
            return {"signals": 0, "trades": 0, "exits": 0}

        # 2. Scan for candidates
        candidates = self.scanner.scan(markets)

        # 3. Poll web agents for intelligence
        web_signal_count = 0
        if self.agents:
            market_queries = [m.question for m in candidates[:20]]
            try:
                web_signals = self.agents.poll_all_sync(market_queries)
                web_signal_count = len(web_signals)
                logger.info(f"Web agents returned {web_signal_count} signals")
            except Exception as e:
                logger.error(f"Web agent polling failed: {e}")

        # 4. Run strategies on candidates (enhanced with web intelligence)
        all_signals = []
        for market in candidates:
            orderbook = self.client.get_orderbook(market.token_id_yes)
            signals = self._run_strategies(market, orderbook)

            # Boost/dampen signals based on web sentiment
            if self.agents:
                self._apply_web_sentiment(signals, market)

            all_signals.extend(signals)

        # 5. Rank signals by weighted score
        ranked = sorted(all_signals, key=lambda s: s.score, reverse=True)
        logger.info(f"Generated {len(ranked)} signals from {len(candidates)} markets (web: {web_signal_count})")

        # 6. Execute top signals
        trades_executed = 0
        for signal in ranked:
            if not self.risk_manager.can_open_position(signal):
                continue

            size_usd = self.risk_manager.calculate_position_size(signal)
            if size_usd == 0:
                continue

            # Place the trade
            trade_result = self._execute_trade(signal, size_usd)
            if trade_result:
                trades_executed += 1
                self.total_trades += 1

                # Notify
                msg = self.notifier.format_trade(
                    "OPEN", signal.market.question, signal.side.value,
                    signal.target_price, size_usd, signal.strategy, signal.edge
                )
                asyncio.get_event_loop().run_until_complete(
                    self.notifier.notify(msg)
                ) if asyncio.get_event_loop().is_running() else None

        # 7. Update existing positions & check exits
        exits = self._check_and_execute_exits()

        elapsed = time.monotonic() - cycle_start
        summary = {
            "cycle": self.cycle_count,
            "markets_scanned": len(markets),
            "candidates": len(candidates),
            "signals": len(ranked),
            "web_signals": web_signal_count,
            "trades": trades_executed,
            "exits": len(exits),
            "positions": len(self.risk_manager.positions),
            "exposure": self.risk_manager.total_exposure,
            "pnl": self.risk_manager.total_pnl,
            "elapsed_s": round(elapsed, 1),
        }

        logger.info(
            f"Cycle {self.cycle_count} complete: "
            f"{len(ranked)} signals, {trades_executed} trades, {len(exits)} exits "
            f"({elapsed:.1f}s)"
        )

        return summary

    def _run_strategies(self, market: Market, orderbook: dict) -> list[Signal]:
        """Run all strategies on a market and return weighted signals."""
        signals = []
        for strategy, weight in self.strategies:
            try:
                signal = strategy.analyze(market, orderbook)
                if signal:
                    # Apply strategy weight to confidence
                    signal.confidence *= weight
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed on {market.question[:40]}: {e}")
        return signals

    def _apply_web_sentiment(self, signals: list[Signal], market: Market):
        """Boost or dampen strategy signals based on web agent intelligence."""
        if not self.agents:
            return

        sentiment_data = self.agents.get_market_sentiment(market.question)
        if sentiment_data["signal_count"] == 0:
            return

        web_sentiment = sentiment_data["sentiment"]
        web_confidence = sentiment_data["confidence"]

        for signal in signals:
            if signal.side == Side.BUY:
                # Buying YES or NO token
                is_yes = signal.token_id == market.token_id_yes

                if is_yes and web_sentiment in (Sentiment.BULLISH, Sentiment.VERY_BULLISH):
                    # Web agrees: YES is likely -> boost
                    boost = 1.0 + (web_confidence * 0.5)
                    signal.confidence *= boost
                    signal.reasoning += f" [WEB: {web_sentiment.value}, boosted {boost:.1f}x]"
                elif is_yes and web_sentiment in (Sentiment.BEARISH, Sentiment.VERY_BEARISH):
                    # Web disagrees: dampen
                    dampen = 1.0 - (web_confidence * 0.4)
                    signal.confidence *= max(0.1, dampen)
                    signal.reasoning += f" [WEB: {web_sentiment.value}, dampened {dampen:.1f}x]"
                elif not is_yes and web_sentiment in (Sentiment.BEARISH, Sentiment.VERY_BEARISH):
                    # Buying NO and web is bearish -> boost
                    boost = 1.0 + (web_confidence * 0.5)
                    signal.confidence *= boost
                    signal.reasoning += f" [WEB: {web_sentiment.value}, boosted {boost:.1f}x]"
                elif not is_yes and web_sentiment in (Sentiment.BULLISH, Sentiment.VERY_BULLISH):
                    # Buying NO but web is bullish -> dampen
                    dampen = 1.0 - (web_confidence * 0.4)
                    signal.confidence *= max(0.1, dampen)
                    signal.reasoning += f" [WEB: {web_sentiment.value}, dampened {dampen:.1f}x]"

    def _execute_trade(self, signal: Signal, size_usd: float) -> bool:
        """Execute a trade from a signal."""
        try:
            # Calculate number of shares
            shares = size_usd / signal.target_price if signal.target_price > 0 else 0
            if shares <= 0:
                return False

            # Place limit order slightly better than target for fill probability
            limit_price = signal.target_price
            result = self.client.place_limit_order(
                token_id=signal.token_id,
                side=signal.side,
                price=limit_price,
                size=shares,
            )

            if result.get("success"):
                position = Position(
                    market_id=signal.market.condition_id,
                    token_id=signal.token_id,
                    question=signal.market.question,
                    side=signal.side,
                    entry_price=signal.target_price,
                    size=shares,
                    cost=size_usd,
                    current_price=signal.target_price,
                    order_id=result.get("orderID", ""),
                )
                self.risk_manager.add_position(position)
                return True
            else:
                logger.warning(f"Trade rejected: {result.get('errorMsg', 'unknown')}")
                return False

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def _check_and_execute_exits(self) -> list[Position]:
        """Update prices and execute exits for stop-loss/take-profit."""
        # Update current prices
        for pos in self.risk_manager.positions:
            try:
                current = self.client.get_price(pos.token_id, pos.side.value)
                if current > 0:
                    pos.current_price = current
            except Exception:
                pass

        # Check for exits
        exits_needed = self.risk_manager.check_exits()
        completed_exits = []

        for pos in exits_needed:
            exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
            result = self.client.place_market_order(
                token_id=pos.token_id,
                side=exit_side,
                amount=pos.size * pos.current_price,
            )
            if result.get("success"):
                self.risk_manager.remove_position(pos.market_id)
                completed_exits.append(pos)

                msg = self.notifier.format_trade(
                    "CLOSE", pos.question, exit_side.value,
                    pos.current_price, pos.cost, "exit",
                    pos.pnl_pct
                )
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self.notifier.notify(msg))
                    else:
                        loop.run_until_complete(self.notifier.notify(msg))
                except RuntimeError:
                    pass

        return completed_exits

    def get_status(self) -> dict:
        """Get full engine status."""
        portfolio = self.risk_manager.get_portfolio_summary()
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        status = {
            "running": self._running,
            "mode": "DRY RUN" if settings.DRY_RUN else "LIVE",
            "uptime_hours": round(uptime / 3600, 1),
            "cycles": self.cycle_count,
            "total_trades": self.total_trades,
            "agents": self.agents.get_agent_status() if self.agents else [],
            "web_signals": self.agents.signal_count if self.agents else 0,
            **portfolio,
        }
        return status
