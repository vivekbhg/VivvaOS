# VivvaOS - Polymarket Trading Bot

Automated prediction market trading bot that scans Polymarket for mispriced markets, generates signals using multiple strategies, and executes trades with built-in risk management. Runs while you sleep.

## How it works

```
Scan Markets → Filter Candidates → Run Strategies → Rank Signals → Execute Trades → Monitor & Exit
     ↑                                                                                    |
     └────────────────────── repeat every N minutes ───────────────────────────────────────┘
```

**Three strategies run in parallel on every market:**

| Strategy | What it does | Edge source |
|----------|-------------|-------------|
| **Value** | Compares current price to orderbook-implied fair value | Book imbalance, VWAP divergence |
| **Momentum** | Detects directional pressure from bid/ask walls | Volume walls, activity ratio |
| **Mispricing** | Finds markets where YES + NO != $1.00 | Underround arb, price skew |

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: add your PRIVATE_KEY (Polygon wallet with USDC)

# 3. Dry run (no real trades, just signals)
python main.py

# 4. Scan markets only
python main.py --scan

# 5. Single cycle
python main.py --once

# 6. Live trading (real money!)
python main.py --live
```

## Configuration

All config via `.env` (see `.env.example`):

```bash
# Core
PRIVATE_KEY=0x...           # Polygon wallet private key
DRY_RUN=true                # true = paper trading, false = real money

# Position sizing
MAX_POSITION_SIZE=50        # Max USD per trade
MAX_TOTAL_EXPOSURE=500      # Max USD across all positions
MIN_EDGE=0.05               # Minimum 5% edge to trade

# Risk management
STOP_LOSS=0.30              # Exit at -30%
TAKE_PROFIT=0.50            # Exit at +50%
MAX_MARKETS=10              # Max simultaneous positions
COOL_DOWN_MINUTES=5         # Minutes between cycles

# Strategy weights (must sum to 1.0)
STRATEGY_VALUE=0.4
STRATEGY_MOMENTUM=0.3
STRATEGY_MISPRICING=0.3
```

## Architecture

```
main.py                     # Entry point & CLI
bot/
  engine.py                 # Core trading loop orchestrator
  models.py                 # Market, Signal, Position data models
  dashboard.py              # Rich CLI display
  strategies/
    base.py                 # Strategy interface
    value.py                # Orderbook value analysis
    momentum.py             # Volume/wall momentum detection
    mispricing.py           # YES+NO arbitrage & skew
  services/
    polymarket_client.py    # Polymarket CLOB API wrapper
    market_scanner.py       # Market filtering & candidate selection
    risk_manager.py         # Position sizing, limits, stop-loss/TP
    notifier.py             # Discord/Telegram alerts
config/
  settings.py               # Environment config loader
```

## Risk Management

- **Position sizing**: Scales with signal confidence and edge magnitude
- **Max exposure**: Hard cap on total USD deployed
- **Stop-loss**: Auto-exits positions that drop below threshold
- **Take-profit**: Auto-exits winners at target
- **No duplicates**: One position per market
- **Market filters**: Skips low-liquidity, wide-spread, and near-certain markets

## Notifications

Set up Discord and/or Telegram alerts for every trade:

```bash
# Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=your_chat_id
```

## Running 24/7

```bash
# Docker
docker build -t vivvaos .
docker run -d --env-file .env --name vivvaos vivvaos

# Or with systemd, pm2, screen, etc.
screen -S vivvaos python main.py --live
```

## Prerequisites

1. Polygon wallet with USDC (for placing trades)
2. Polymarket account connected to that wallet
3. USDC approved for Polymarket's exchange contracts

## Disclaimer

This bot trades real money on prediction markets. Use at your own risk. Start with DRY_RUN=true to understand the signals before going live. Past performance of any strategy does not guarantee future results.
