"""Wrapper around the Polymarket CLOB API client."""
import logging
import time

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType

from config.settings import settings
from bot.models import Market, Side

logger = logging.getLogger(__name__)


class PolymarketClient:
    """High-level interface to Polymarket CLOB API."""

    def __init__(self):
        self.client = ClobClient(
            host=settings.CLOB_API_URL,
            key=settings.PRIVATE_KEY,
            chain_id=settings.CHAIN_ID,
        )
        self._creds = None

    def authenticate(self):
        """Derive or create API credentials."""
        try:
            self._creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(self._creds)
            logger.info("Authenticated with Polymarket CLOB")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def get_markets(self) -> list[Market]:
        """Fetch all active markets and return parsed Market objects."""
        try:
            markets = []
            next_cursor = "MA=="
            # Paginate through all markets (API returns {data: [...], next_cursor: "..."})
            while next_cursor:
                resp = self.client.get_markets(next_cursor=next_cursor)

                # Handle both list and paginated dict responses
                if isinstance(resp, dict):
                    raw_list = resp.get("data", [])
                    next_cursor = resp.get("next_cursor", "")
                    # Empty or "LTE=" cursor means no more pages
                    if not next_cursor or next_cursor == "LTE=":
                        next_cursor = ""
                elif isinstance(resp, list):
                    raw_list = resp
                    next_cursor = ""
                else:
                    logger.warning(f"Unexpected markets response type: {type(resp)}")
                    break

                for m in raw_list:
                    if not isinstance(m, dict):
                        continue
                    try:
                        markets.append(self._parse_market(m))
                    except (KeyError, ValueError, IndexError):
                        continue

            logger.info(f"Fetched {len(markets)} markets")
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    def _parse_market(self, raw: dict) -> Market:
        """Parse raw market data into Market model."""
        tokens = raw.get("tokens", [])
        if len(tokens) < 2:
            raise ValueError("Market needs at least 2 tokens")

        yes_token = tokens[0]
        no_token = tokens[1]

        yes_price = float(yes_token.get("price", 0.5))
        no_price = float(no_token.get("price", 0.5))

        return Market(
            condition_id=raw.get("condition_id", ""),
            question=raw.get("question", "Unknown"),
            token_id_yes=yes_token.get("token_id", ""),
            token_id_no=no_token.get("token_id", ""),
            yes_price=yes_price,
            no_price=no_price,
            spread=abs(yes_price - (1 - no_price)),
            volume_24h=float(raw.get("volume_num_24hr", 0)),
            liquidity=float(raw.get("liquidity", 0)),
            end_date=raw.get("end_date_iso", ""),
            category=raw.get("category", ""),
        )

    def get_orderbook(self, token_id: str) -> dict:
        """Get orderbook for a token."""
        try:
            book = self.client.get_order_book(token_id)
            return book
        except Exception as e:
            logger.error(f"Failed to get orderbook for {token_id}: {e}")
            return {"bids": [], "asks": []}

    def get_price(self, token_id: str, side: str = "BUY") -> float:
        """Get current price for a token."""
        try:
            resp = self.client.get_price(token_id, side)
            return float(resp.get("price", 0))
        except Exception as e:
            logger.error(f"Failed to get price for {token_id}: {e}")
            return 0.0

    def get_midpoint(self, token_id: str) -> float:
        """Get midpoint price."""
        try:
            resp = self.client.get_midpoint(token_id)
            return float(resp.get("mid", 0))
        except Exception as e:
            logger.error(f"Failed to get midpoint for {token_id}: {e}")
            return 0.0

    def place_limit_order(
        self, token_id: str, side: Side, price: float, size: float
    ) -> dict:
        """Place a limit order (GTC)."""
        if settings.DRY_RUN:
            logger.info(f"[DRY RUN] Limit {side.value} {size:.1f} @ ${price:.3f} on {token_id[:12]}...")
            return {"success": True, "orderID": f"dry_run_{int(time.time())}", "dry_run": True}

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.value,
            )
            signed = self.client.create_order(order_args)
            resp = self.client.post_order(signed, OrderType.GTC)
            logger.info(f"Order placed: {side.value} {size:.1f} @ ${price:.3f} -> {resp}")
            return resp
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return {"success": False, "errorMsg": str(e)}

    def place_market_order(
        self, token_id: str, side: Side, amount: float
    ) -> dict:
        """Place a market order (FOK)."""
        if settings.DRY_RUN:
            logger.info(f"[DRY RUN] Market {side.value} ${amount:.2f} on {token_id[:12]}...")
            return {"success": True, "orderID": f"dry_run_{int(time.time())}", "dry_run": True}

        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side.value,
            )
            signed = self.client.create_market_order(order_args)
            resp = self.client.post_order(signed, OrderType.FOK)
            logger.info(f"Market order: {side.value} ${amount:.2f} -> {resp}")
            return resp
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return {"success": False, "errorMsg": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if settings.DRY_RUN:
            logger.info(f"[DRY RUN] Cancel order {order_id}")
            return True
        try:
            self.client.cancel(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return False

    def cancel_all(self) -> bool:
        """Cancel all active orders."""
        if settings.DRY_RUN:
            logger.info("[DRY RUN] Cancel all orders")
            return True
        try:
            self.client.cancel_all()
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False
