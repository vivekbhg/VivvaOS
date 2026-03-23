import time
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.user import ApiKey, UsageLog


async def get_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    result = await db.execute(select(ApiKey).where(ApiKey.key == x_api_key, ApiKey.is_active == True))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Reset daily counter if needed
    now = datetime.now(timezone.utc)
    if api_key.last_reset.date() < now.date():
        api_key.requests_today = 0
        api_key.last_reset = now

    # Load user to check plan
    from app.models.user import User
    user_result = await db.execute(select(User).where(User.id == api_key.user_id))
    user = user_result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    # Check rate limit
    limit = settings.RATE_LIMITS.get(user.plan, 100)
    if api_key.requests_today >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {limit} requests exceeded. Upgrade your plan at {settings.BASE_URL}/pricing",
        )

    # Increment counter
    api_key.requests_today += 1
    await db.commit()

    # Store user plan on the key object for endpoint access
    api_key._user_plan = user.plan
    return api_key


async def log_usage(api_key_id: int, endpoint: str, response_ms: int, db: AsyncSession):
    log = UsageLog(api_key_id=api_key_id, endpoint=endpoint, response_ms=response_ms)
    db.add(log)
    await db.commit()
