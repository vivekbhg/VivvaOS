import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-me")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./vivvaapi.db")
    BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8000")

    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRO_PRICE_ID: str = os.getenv("STRIPE_PRO_PRICE_ID", "")
    STRIPE_BUSINESS_PRICE_ID: str = os.getenv("STRIPE_BUSINESS_PRICE_ID", "")

    # Rate limits per day by plan
    RATE_LIMITS = {
        "free": 100,
        "pro": 10_000,
        "business": 100_000,
    }

    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


settings = Settings()
