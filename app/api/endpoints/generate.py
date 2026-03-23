"""Generation API endpoints - QR codes, UUIDs, passwords, etc."""
import io
import secrets
import string
import time
import uuid
from base64 import b64encode

import qrcode
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_api_key, log_usage
from app.core.database import get_db
from app.models.user import ApiKey

router = APIRouter(prefix="/generate", tags=["Generators"])


class QrInput(BaseModel):
    data: str = Field(..., max_length=4000)
    size: int = Field(default=10, ge=1, le=40)


class PasswordInput(BaseModel):
    length: int = Field(default=16, ge=8, le=128)
    uppercase: bool = True
    lowercase: bool = True
    digits: bool = True
    special: bool = True
    count: int = Field(default=1, ge=1, le=50)


class UuidInput(BaseModel):
    count: int = Field(default=1, ge=1, le=100)
    version: int = Field(default=4, ge=1, le=4)


@router.post("/qr-code")
async def generate_qr(
    body: QrInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Generate a QR code as base64 PNG."""
    start = time.monotonic()
    qr = qrcode.QRCode(version=1, box_size=body.size, border=2)
    qr.add_data(body.data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = b64encode(buf.getvalue()).decode()

    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/generate/qr-code", elapsed, db)
    return {
        "data": {
            "base64_png": b64,
            "data_uri": f"data:image/png;base64,{b64}",
        },
        "processing_ms": elapsed,
    }


@router.post("/password")
async def generate_password(
    body: PasswordInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Generate cryptographically secure random passwords."""
    start = time.monotonic()
    charset = ""
    if body.uppercase:
        charset += string.ascii_uppercase
    if body.lowercase:
        charset += string.ascii_lowercase
    if body.digits:
        charset += string.digits
    if body.special:
        charset += string.punctuation
    if not charset:
        charset = string.ascii_letters + string.digits

    passwords = ["".join(secrets.choice(charset) for _ in range(body.length)) for _ in range(body.count)]

    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/generate/password", elapsed, db)
    return {"data": {"passwords": passwords}, "processing_ms": elapsed}


@router.post("/uuid")
async def generate_uuid(
    body: UuidInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Generate UUIDs."""
    start = time.monotonic()
    uuids = [str(uuid.uuid4()) for _ in range(body.count)]
    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/generate/uuid", elapsed, db)
    return {"data": {"uuids": uuids}, "processing_ms": elapsed}
