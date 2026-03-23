"""Authentication & account management routes."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token, generate_api_key
from app.models.user import User, ApiKey

router = APIRouter(prefix="/auth", tags=["Authentication"])


class RegisterInput(BaseModel):
    email: EmailStr
    password: str


class LoginInput(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
async def register(body: RegisterInput, db: AsyncSession = Depends(get_db)):
    """Register a new account and get an API key."""
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    await db.flush()

    key = ApiKey(key=generate_api_key(), user_id=user.id, name="Default")
    db.add(key)
    await db.commit()

    token = create_access_token({"sub": str(user.id)})
    return {
        "message": "Account created! Save your API key - you won't see it again.",
        "api_key": key.key,
        "token": token,
        "plan": "free",
    }


@router.post("/login")
async def login(body: LoginInput, db: AsyncSession = Depends(get_db)):
    """Login and get a session token."""
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({"sub": str(user.id)})

    # Fetch API keys
    keys_result = await db.execute(select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.is_active == True))
    keys = keys_result.scalars().all()

    return {
        "token": token,
        "plan": user.plan,
        "api_keys": [{"name": k.name, "key": k.key[:8] + "...", "requests_today": k.requests_today} for k in keys],
    }


@router.post("/api-keys")
async def create_api_key(
    name: str = "New Key",
    token: str = "",
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key (requires token from login)."""
    from jose import jwt, JWTError
    from app.core.config import settings

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(401, "Invalid token")

    key = ApiKey(key=generate_api_key(), user_id=user_id, name=name)
    db.add(key)
    await db.commit()
    return {"api_key": key.key, "name": key.name}
