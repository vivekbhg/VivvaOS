"""Tests for VivvaAPI endpoints."""
import uuid

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.database import engine, Base


@pytest.fixture(autouse=True)
async def reset_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def api_key(client: AsyncClient):
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post("/auth/register", json={"email": email, "password": "test1234"})
    assert res.status_code == 200
    return res.json()["api_key"]


@pytest.mark.anyio
async def test_health(client):
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.anyio
async def test_register_and_login(client):
    res = await client.post("/auth/register", json={"email": "new@example.com", "password": "pass1234"})
    assert res.status_code == 200
    data = res.json()
    assert "api_key" in data
    assert data["api_key"].startswith("vv_")

    res = await client.post("/auth/login", json={"email": "new@example.com", "password": "pass1234"})
    assert res.status_code == 200
    assert "token" in res.json()


@pytest.mark.anyio
async def test_text_analyze(client, api_key):
    res = await client.post(
        "/api/v1/text/analyze",
        json={"text": "Hello world. This is a test sentence."},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["words"] == 7
    assert data["sentences"] == 2


@pytest.mark.anyio
async def test_text_hash(client, api_key):
    res = await client.post(
        "/api/v1/text/hash",
        json={"text": "hello"},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    assert "sha256" in res.json()["data"]


@pytest.mark.anyio
async def test_text_slugify(client, api_key):
    res = await client.post(
        "/api/v1/text/slugify",
        json={"text": "Hello World! This is a Test"},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    assert res.json()["data"]["slug"] == "hello-world-this-is-a-test"


@pytest.mark.anyio
async def test_text_extract_emails(client, api_key):
    res = await client.post(
        "/api/v1/text/extract",
        json={"text": "Contact us at hello@example.com or support@test.org", "extract": "emails"},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    assert res.json()["data"]["count"] == 2


@pytest.mark.anyio
async def test_data_base64(client, api_key):
    res = await client.post(
        "/api/v1/data/base64",
        json={"text": "Hello World", "action": "encode"},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    encoded = res.json()["data"]["result"]

    res = await client.post(
        "/api/v1/data/base64",
        json={"encoded": encoded, "action": "decode"},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    assert res.json()["data"]["result"] == "Hello World"


@pytest.mark.anyio
async def test_data_json_diff(client, api_key):
    res = await client.post(
        "/api/v1/data/json-diff",
        json={"a": {"name": "old", "value": 1}, "b": {"name": "new", "value": 1}},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    assert res.json()["data"]["total_changes"] == 1


@pytest.mark.anyio
async def test_generate_password(client, api_key):
    res = await client.post(
        "/api/v1/generate/password",
        json={"length": 20, "count": 3},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    passwords = res.json()["data"]["passwords"]
    assert len(passwords) == 3
    assert all(len(p) == 20 for p in passwords)


@pytest.mark.anyio
async def test_generate_uuid(client, api_key):
    res = await client.post(
        "/api/v1/generate/uuid",
        json={"count": 5},
        headers={"X-API-Key": api_key},
    )
    assert res.status_code == 200
    assert len(res.json()["data"]["uuids"]) == 5


@pytest.mark.anyio
async def test_invalid_api_key(client):
    res = await client.post(
        "/api/v1/text/analyze",
        json={"text": "test"},
        headers={"X-API-Key": "invalid_key"},
    )
    assert res.status_code == 401


@pytest.mark.anyio
async def test_missing_api_key(client):
    res = await client.post("/api/v1/text/analyze", json={"text": "test"})
    assert res.status_code == 422
