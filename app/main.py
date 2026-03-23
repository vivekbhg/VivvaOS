"""VivvaAPI - Paid developer utility API service."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.database import init_db
from app.api.endpoints import text, data, generate, auth_routes, billing


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="VivvaAPI",
    description="Developer utility API - text processing, data transformation, and generators.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

templates = Jinja2Templates(directory="app/templates")

# Mount API routes under /api/v1
app.include_router(text.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(generate.router, prefix="/api/v1")

# Auth & billing at root level
app.include_router(auth_routes.router)
app.include_router(billing.router)


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "base_url": settings.BASE_URL})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VivvaAPI"}
