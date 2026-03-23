#!/usr/bin/env python3
"""VivvaOS - Polymarket Trading Bot

Usage:
    python main.py              # Run the bot (dry run by default)
    python main.py --live       # Run with live trading (careful!)
    python main.py --once       # Run a single cycle then exit
    python main.py --scan       # Scan markets only (no trading)
    python main.py --no-agents  # Disable web monitoring agents
"""
import argparse
import logging
import signal
import sys
import time

from config.settings import settings
from bot.engine import TradingEngine
from bot.dashboard import (
    console, print_banner, print_status, print_portfolio,
    print_cycle_summary,
)


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def scan_only(engine: TradingEngine):
    """Just scan markets and show opportunities without trading."""
    console.print("\n[bold]Scanning markets...[/bold]\n")

    markets = engine.client.get_markets()
    candidates = engine.scanner.scan(markets)

    from rich.table import Table
    table = Table(title=f"Market Candidates ({len(candidates)})", border_style="blue")
    table.add_column("#", justify="right", width=4)
    table.add_column("Market", max_width=55)
    table.add_column("YES", justify="right")
    table.add_column("NO", justify="right")
    table.add_column("Spread", justify="right")
    table.add_column("Volume 24h", justify="right")
    table.add_column("Liquidity", justify="right")

    for i, m in enumerate(candidates[:30], 1):
        table.add_row(
            str(i),
            m.question[:55],
            f"${m.yes_price:.3f}",
            f"${m.no_price:.3f}",
            f"${m.spread:.3f}",
            f"${m.volume_24h:,.0f}",
            f"${m.liquidity:,.0f}",
        )

    console.print(table)

    # Run strategies on top candidates
    console.print(f"\n[bold]Analyzing top {min(10, len(candidates))} markets...[/bold]\n")
    all_signals = []
    for market in candidates[:10]:
        orderbook = engine.client.get_orderbook(market.token_id_yes)
        for strategy, weight in engine.strategies:
            sig = strategy.analyze(market, orderbook)
            if sig:
                sig.confidence *= weight
                all_signals.append(sig)

    if all_signals:
        all_signals.sort(key=lambda s: s.score, reverse=True)
        sig_table = Table(title="Trading Signals", border_style="yellow")
        sig_table.add_column("Market", max_width=40)
        sig_table.add_column("Strategy")
        sig_table.add_column("Side")
        sig_table.add_column("Edge", justify="right")
        sig_table.add_column("Confidence", justify="right")
        sig_table.add_column("Reasoning", max_width=50)

        for s in all_signals[:15]:
            sig_table.add_row(
                s.market.question[:40],
                s.strategy,
                s.side.value,
                f"{s.edge:.1%}",
                f"{s.confidence:.1%}",
                s.reasoning[:50],
            )
        console.print(sig_table)
    else:
        console.print("[dim]No signals found[/dim]")


def main():
    parser = argparse.ArgumentParser(description="VivvaOS Polymarket Trading Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (override DRY_RUN)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--scan", action="store_true", help="Scan markets only")
    parser.add_argument("--no-agents", action="store_true", help="Disable web monitoring agents")
    args = parser.parse_args()

    setup_logging()
    print_banner()

    # Validate config
    if args.live:
        settings.DRY_RUN = False

    errors = settings.validate()
    if errors:
        console.print("[red bold]Configuration errors:[/red bold]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        console.print("\nCopy .env.example to .env and fill in your values.")
        sys.exit(1)

    if not settings.DRY_RUN:
        console.print("\n[red bold]*** LIVE TRADING MODE ***[/red bold]")
        console.print("[red]Real money will be used. Press Ctrl+C within 5s to cancel.[/red]\n")
        time.sleep(5)

    # Initialize engine
    engine = TradingEngine(enable_agents=not args.no_agents)
    engine.start()

    # Handle graceful shutdown
    def shutdown(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        engine.stop()
        print_status(engine)
        print_portfolio(engine)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Scan-only mode
    if args.scan:
        scan_only(engine)
        return

    # Main loop
    while True:
        try:
            summary = engine.run_cycle()
            print_cycle_summary(summary)
            print_portfolio(engine)

            if args.once:
                break

            console.print(
                f"[dim]Next cycle in {settings.COOL_DOWN_MINUTES}m... "
                f"(Ctrl+C to stop)[/dim]\n"
            )
            time.sleep(settings.COOL_DOWN_MINUTES * 60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Cycle error: {e}[/red]")
            logging.exception("Cycle error")
            time.sleep(30)

    # Final status
    print_status(engine)
    print_portfolio(engine)


if __name__ == "__main__":
    main()
