"""Bot configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Polymarket connection
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    CLOB_API_URL: str = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
    CHAIN_ID: int = int(os.getenv("CHAIN_ID", "137"))

    # Trading parameters
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "50"))
    MAX_TOTAL_EXPOSURE: float = float(os.getenv("MAX_TOTAL_EXPOSURE", "500"))
    MIN_EDGE: float = float(os.getenv("MIN_EDGE", "0.05"))
    MIN_LIQUIDITY: float = float(os.getenv("MIN_LIQUIDITY", "1000"))
    MAX_SPREAD: float = float(os.getenv("MAX_SPREAD", "0.10"))

    # Risk management
    STOP_LOSS: float = float(os.getenv("STOP_LOSS", "0.30"))
    TAKE_PROFIT: float = float(os.getenv("TAKE_PROFIT", "0.50"))
    MAX_MARKETS: int = int(os.getenv("MAX_MARKETS", "10"))
    COOL_DOWN_MINUTES: int = int(os.getenv("COOL_DOWN_MINUTES", "5"))

    # Strategy weights
    STRATEGY_VALUE: float = float(os.getenv("STRATEGY_VALUE", "0.4"))
    STRATEGY_MOMENTUM: float = float(os.getenv("STRATEGY_MOMENTUM", "0.3"))
    STRATEGY_MISPRICING: float = float(os.getenv("STRATEGY_MISPRICING", "0.3"))

    # Notifications
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Operations
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"

    def validate(self) -> list[str]:
        """Return list of config errors."""
        errors = []
        if not self.PRIVATE_KEY or self.PRIVATE_KEY == "0x_your_private_key_here":
            errors.append("PRIVATE_KEY not set")
        if self.MAX_POSITION_SIZE <= 0:
            errors.append("MAX_POSITION_SIZE must be positive")
        if self.MAX_TOTAL_EXPOSURE < self.MAX_POSITION_SIZE:
            errors.append("MAX_TOTAL_EXPOSURE must be >= MAX_POSITION_SIZE")
        weights = self.STRATEGY_VALUE + self.STRATEGY_MOMENTUM + self.STRATEGY_MISPRICING
        if abs(weights - 1.0) > 0.01:
            errors.append(f"Strategy weights must sum to 1.0, got {weights}")
        return errors


settings = Settings()
