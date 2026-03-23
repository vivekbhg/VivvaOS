"""Data transformation API endpoints."""
import base64
import csv
import io
import json
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_api_key, log_usage
from app.core.database import get_db
from app.models.user import ApiKey

router = APIRouter(prefix="/data", tags=["Data Tools"])


class JsonInput(BaseModel):
    data: dict | list = Field(...)


class CsvInput(BaseModel):
    csv_text: str = Field(..., max_length=500_000)
    delimiter: str = Field(default=",", max_length=1)


class Base64Input(BaseModel):
    text: str = Field(default="", max_length=500_000)
    encoded: str = Field(default="", max_length=500_000)
    action: str = Field(default="encode", description="encode or decode")


class JsonDiffInput(BaseModel):
    a: dict | list
    b: dict | list


@router.post("/csv-to-json")
async def csv_to_json(
    body: CsvInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Convert CSV text to JSON array."""
    start = time.monotonic()
    reader = csv.DictReader(io.StringIO(body.csv_text), delimiter=body.delimiter)
    rows = list(reader)
    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/data/csv-to-json", elapsed, db)
    return {"data": {"rows": rows, "count": len(rows)}, "processing_ms": elapsed}


@router.post("/json-to-csv")
async def json_to_csv(
    body: JsonInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Convert JSON array to CSV text."""
    start = time.monotonic()
    if not isinstance(body.data, list) or not body.data:
        raise HTTPException(400, "Input must be a non-empty JSON array of objects")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=body.data[0].keys())
    writer.writeheader()
    writer.writerows(body.data)
    csv_text = output.getvalue()

    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/data/json-to-csv", elapsed, db)
    return {"data": {"csv": csv_text, "rows": len(body.data)}, "processing_ms": elapsed}


@router.post("/base64")
async def base64_convert(
    body: Base64Input,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Encode or decode Base64."""
    start = time.monotonic()
    if body.action == "encode":
        result = base64.b64encode(body.text.encode()).decode()
    elif body.action == "decode":
        try:
            result = base64.b64decode(body.encoded).decode()
        except Exception:
            raise HTTPException(400, "Invalid base64 input")
    else:
        raise HTTPException(400, "action must be 'encode' or 'decode'")

    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/data/base64", elapsed, db)
    return {"data": {"result": result}, "processing_ms": elapsed}


@router.post("/json-diff")
async def json_diff(
    body: JsonDiffInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Compare two JSON objects and return differences."""
    start = time.monotonic()

    def diff(a, b, path=""):
        changes = []
        if isinstance(a, dict) and isinstance(b, dict):
            for key in set(list(a.keys()) + list(b.keys())):
                p = f"{path}.{key}" if path else key
                if key not in a:
                    changes.append({"path": p, "type": "added", "value": b[key]})
                elif key not in b:
                    changes.append({"path": p, "type": "removed", "value": a[key]})
                else:
                    changes.extend(diff(a[key], b[key], p))
        elif a != b:
            changes.append({"path": path or "$", "type": "changed", "old": a, "new": b})
        return changes

    changes = diff(body.a, body.b)
    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/data/json-diff", elapsed, db)
    return {"data": {"changes": changes, "total_changes": len(changes), "identical": len(changes) == 0}, "processing_ms": elapsed}


@router.post("/flatten-json")
async def flatten_json(
    body: JsonInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Flatten a nested JSON object to dot-notation keys."""
    start = time.monotonic()

    def flatten(obj, prefix=""):
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                items.update(flatten(v, new_key))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                items.update(flatten(v, f"{prefix}[{i}]"))
        else:
            items[prefix] = obj
        return items

    result = flatten(body.data)
    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/data/flatten-json", elapsed, db)
    return {"data": result, "processing_ms": elapsed}
