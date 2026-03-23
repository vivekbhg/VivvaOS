"""Send notifications via Discord and/or Telegram."""
import logging

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class Notifier:
    """Sends trade notifications to Discord/Telegram."""

    async def notify(self, message: str, level: str = "info"):
        """Send notification to all configured channels."""
        if settings.DISCORD_WEBHOOK_URL:
            await self._send_discord(message)
        if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
            await self._send_telegram(message)

    async def _send_discord(self, message: str):
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    settings.DISCORD_WEBHOOK_URL,
                    json={"content": message[:2000]},
                    timeout=10,
                )
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")

    async def _send_telegram(self, message: str):
        try:
            url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as client:
                await client.post(
                    url,
                    json={
                        "chat_id": settings.TELEGRAM_CHAT_ID,
                        "text": message[:4000],
                        "parse_mode": "Markdown",
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")

    def format_trade(self, action: str, question: str, side: str,
                     price: float, size: float, strategy: str, edge: float) -> str:
        """Format a trade notification message."""
        emoji = "BUY" if action == "OPEN" else "CLOSE"
        return (
            f"[{emoji}] {action} | {strategy}\n"
            f"Market: {question[:80]}\n"
            f"Side: {side} @ ${price:.3f} | Size: ${size:.2f}\n"
            f"Edge: {edge:.1%}"
        )

    def format_portfolio(self, summary: dict) -> str:
        """Format portfolio summary."""
        return (
            f"Portfolio Update\n"
            f"Positions: {summary['positions']} | "
            f"Exposure: ${summary['total_exposure']:.2f}\n"
            f"P&L: ${summary['total_pnl']:+.2f} ({summary['pnl_pct']:+.1%})"
        )
