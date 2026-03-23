import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import settings


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${h.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    salt, stored_hash = hashed.split("$", 1)
    h = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), 100_000)
    return hmac.compare_digest(h.hex(), stored_hash)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def generate_api_key() -> str:
    return f"vv_{secrets.token_urlsafe(32)}"
