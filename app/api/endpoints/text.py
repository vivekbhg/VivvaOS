"""Text processing API endpoints - the bread and butter of the service."""
import hashlib
import re
import time
from collections import Counter

import bleach
import markdown
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_api_key, log_usage
from app.core.database import get_db
from app.models.user import ApiKey

router = APIRouter(prefix="/text", tags=["Text Processing"])


class TextInput(BaseModel):
    text: str = Field(..., max_length=100_000)


class MarkdownInput(BaseModel):
    markdown: str = Field(..., max_length=100_000)
    sanitize: bool = True


class SummarizeInput(BaseModel):
    text: str = Field(..., max_length=50_000)
    max_sentences: int = Field(default=3, ge=1, le=10)


class ExtractInput(BaseModel):
    text: str = Field(..., max_length=100_000)
    extract: str = Field(..., description="What to extract: emails, urls, phones, hashtags, mentions")


@router.post("/analyze")
async def analyze_text(
    body: TextInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Analyze text: word count, character count, reading time, sentiment estimate, etc."""
    start = time.monotonic()
    text = body.text

    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    word_freq = Counter(w.lower().strip(".,!?;:") for w in words)

    result = {
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "")),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 1),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        "reading_time_seconds": round(len(words) / 4.2),  # ~250 wpm
        "top_words": word_freq.most_common(10),
        "unique_words": len(set(w.lower() for w in words)),
    }

    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/text/analyze", elapsed, db)
    return {"data": result, "processing_ms": elapsed}


@router.post("/markdown-to-html")
async def markdown_to_html(
    body: MarkdownInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Convert Markdown to sanitized HTML."""
    start = time.monotonic()
    html = markdown.markdown(body.markdown, extensions=["tables", "fenced_code", "toc"])
    if body.sanitize:
        html = bleach.clean(
            html,
            tags=["h1","h2","h3","h4","h5","h6","p","a","img","ul","ol","li",
                  "code","pre","blockquote","table","thead","tbody","tr","th","td",
                  "strong","em","br","hr","div","span"],
            attributes={"a": ["href", "title"], "img": ["src", "alt"]},
        )

    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/text/markdown-to-html", elapsed, db)
    return {"data": {"html": html}, "processing_ms": elapsed}


@router.post("/extract")
async def extract_patterns(
    body: ExtractInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Extract emails, URLs, phone numbers, hashtags, or mentions from text."""
    start = time.monotonic()
    patterns = {
        "emails": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "urls": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "phones": r'[\+]?[(]?[0-9]{1,4}[)]?[-\s./0-9]{7,15}',
        "hashtags": r'#[a-zA-Z0-9_]+',
        "mentions": r'@[a-zA-Z0-9_]+',
    }
    if body.extract not in patterns:
        raise HTTPException(400, f"Invalid extract type. Choose from: {list(patterns.keys())}")

    matches = re.findall(patterns[body.extract], body.text)
    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/text/extract", elapsed, db)
    return {"data": {"type": body.extract, "matches": list(set(matches)), "count": len(set(matches))}, "processing_ms": elapsed}


@router.post("/hash")
async def hash_text(
    body: TextInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Generate multiple hash digests of the input text."""
    start = time.monotonic()
    text_bytes = body.text.encode("utf-8")
    result = {
        "md5": hashlib.md5(text_bytes).hexdigest(),
        "sha1": hashlib.sha1(text_bytes).hexdigest(),
        "sha256": hashlib.sha256(text_bytes).hexdigest(),
        "sha512": hashlib.sha512(text_bytes).hexdigest(),
    }
    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/text/hash", elapsed, db)
    return {"data": result, "processing_ms": elapsed}


@router.post("/slugify")
async def slugify_text(
    body: TextInput,
    api_key: ApiKey = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Convert text to a URL-friendly slug."""
    start = time.monotonic()
    slug = re.sub(r'[^\w\s-]', '', body.text.lower())
    slug = re.sub(r'[-\s]+', '-', slug).strip('-')
    elapsed = int((time.monotonic() - start) * 1000)
    await log_usage(api_key.id, "/text/slugify", elapsed, db)
    return {"data": {"slug": slug}, "processing_ms": elapsed}
