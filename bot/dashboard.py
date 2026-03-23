"""Rich CLI dashboard for monitoring the bot."""
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from bot.engine import TradingEngine

console = Console()
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner."""
    banner = """
 ██╗   ██╗██╗██╗   ██╗██╗   ██╗ █████╗  ██████╗ ███████╗
 ██║   ██║██║██║   ██║██║   ██║██╔══██╗██╔═══██╗██╔════╝
 ██║   ██║██║██║   ██║██║   ██║███████║██║   ██║███████╗
 ╚██╗ ██╔╝██║╚██╗ ██╔╝╚██╗ ██╔╝██╔══██║██║   ██║╚════██║
  ╚████╔╝ ██║ ╚████╔╝  ╚████╔╝ ██║  ██║╚██████╔╝███████║
   ╚═══╝  ╚═╝  ╚═══╝    ╚═══╝  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
                Polymarket Trading Bot
    """
    console.print(banner, style="bold magenta")


def print_status(engine: TradingEngine):
    """Print current engine status."""
    status = engine.get_status()

    # Status panel
    mode_color = "yellow" if status["mode"] == "DRY RUN" else "red bold"
    status_text = Text()
    status_text.append(f"Mode: ", style="dim")
    status_text.append(f"{status['mode']}\n", style=mode_color)
    status_text.append(f"Uptime: ", style="dim")
    status_text.append(f"{status['uptime_hours']}h\n")
    status_text.append(f"Cycles: ", style="dim")
    status_text.append(f"{status['cycles']}\n")
    status_text.append(f"Total Trades: ", style="dim")
    status_text.append(f"{status['total_trades']}\n")
    status_text.append(f"Web Signals: ", style="dim")
    status_text.append(f"{status.get('web_signals', 0)}\n")

    # Agent status
    agents = status.get("agents", [])
    if agents:
        status_text.append(f"\nWeb Agents:\n", style="bold")
        for agent in agents:
            status_text.append(f"  {agent['name']}: ", style="dim")
            status_text.append(f"{agent['signal_count']} signals (last: {agent['last_run']})\n")

    console.print(Panel(status_text, title="Bot Status", border_style="blue"))


def print_portfolio(engine: TradingEngine):
    """Print portfolio table."""
    rm = engine.risk_manager

    if not rm.positions:
        console.print("[dim]No open positions[/dim]\n")
        return

    table = Table(title="Open Positions", border_style="blue")
    table.add_column("Market", max_width=45)
    table.add_column("Side", justify="center")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Size ($)", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")

    for pos in rm.positions:
        pnl_color = "green" if pos.pnl >= 0 else "red"
        table.add_row(
            pos.question[:45],
            pos.side.value,
            f"${pos.entry_price:.3f}",
            f"${pos.current_price:.3f}",
            f"${pos.cost:.2f}",
            f"[{pnl_color}]${pos.pnl:+.2f}[/{pnl_color}]",
            f"[{pnl_color}]{pos.pnl_pct:+.1%}[/{pnl_color}]",
        )

    # Summary row
    total_pnl = rm.total_pnl
    total_color = "green" if total_pnl >= 0 else "red"
    table.add_row(
        "[bold]TOTAL[/bold]", "", "", "",
        f"[bold]${rm.total_exposure:.2f}[/bold]",
        f"[bold {total_color}]${total_pnl:+.2f}[/bold {total_color}]",
        f"[bold {total_color}]{total_pnl / max(rm.total_exposure, 1):+.1%}[/bold {total_color}]",
    )

    console.print(table)
    console.print()


def print_cycle_summary(summary: dict):
    """Print summary of a trading cycle."""
    table = Table(show_header=False, border_style="dim")
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Markets Scanned", str(summary["markets_scanned"]))
    table.add_row("Candidates", str(summary["candidates"]))
    table.add_row("Signals", str(summary["signals"]))
    table.add_row("Web Signals", str(summary.get("web_signals", 0)))
    table.add_row("Trades Executed", str(summary["trades"]))
    table.add_row("Exits", str(summary["exits"]))
    table.add_row("Open Positions", str(summary["positions"]))

    pnl_color = "green" if summary["pnl"] >= 0 else "red"
    table.add_row("Exposure", f"${summary['exposure']:.2f}")
    table.add_row("P&L", f"[{pnl_color}]${summary['pnl']:+.2f}[/{pnl_color}]")
    table.add_row("Cycle Time", f"{summary['elapsed_s']}s")

    console.print(Panel(table, title=f"Cycle {summary['cycle']}", border_style="green"))


def print_signals(signals: list):
    """Print ranked signals."""
    if not signals:
        return

    table = Table(title="Top Signals", border_style="yellow")
    table.add_column("Market", max_width=40)
    table.add_column("Strategy")
    table.add_column("Side", justify="center")
    table.add_column("Edge", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Score", justify="right")

    for s in signals[:10]:
        table.add_row(
            s.market.question[:40],
            s.strategy,
            s.side.value,
            f"{s.edge:.1%}",
            f"{s.confidence:.1%}",
            f"{s.score:.3f}",
        )

    console.print(table)
