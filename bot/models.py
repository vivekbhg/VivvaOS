"""Data models for the trading bot."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalStrength(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Market:
    """A Polymarket prediction market."""
    condition_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    yes_price: float
    no_price: float
    spread: float
    volume_24h: float
    liquidity: float
    end_date: str = ""
    category: str = ""

    @property
    def midpoint(self) -> float:
        return (self.yes_price + (1 - self.no_price)) / 2

    @property
    def implied_probability(self) -> float:
        return self.yes_price


@dataclass
class Signal:
    """A trading signal from a strategy."""
    market: Market
    strategy: str
    side: Side
    token_id: str
    target_price: float
    edge: float
    confidence: float  # 0-1
    strength: SignalStrength
    reasoning: str

    @property
    def score(self) -> float:
        return self.edge * self.confidence


@dataclass
class Position:
    """An active position."""
    market_id: str
    token_id: str
    question: str
    side: Side
    entry_price: float
    size: float  # number of shares
    cost: float  # USD spent
    current_price: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: str = ""

    @property
    def pnl(self) -> float:
        if self.side == Side.BUY:
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def pnl_pct(self) -> float:
        if self.cost == 0:
            return 0.0
        return self.pnl / self.cost

    @property
    def should_stop_loss(self) -> bool:
        from config.settings import settings
        return self.pnl_pct <= -settings.STOP_LOSS

    @property
    def should_take_profit(self) -> bool:
        from config.settings import settings
        return self.pnl_pct >= settings.TAKE_PROFIT


@dataclass
class TradeRecord:
    """Completed trade record for tracking P&L."""
    market_id: str
    question: str
    side: Side
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    strategy: str
    opened_at: datetime
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
