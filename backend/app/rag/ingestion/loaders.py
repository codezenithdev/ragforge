"""Document loaders.

Each loader returns ``list[dict]`` of content blocks with source metadata
preserved (page number / section header / URL). The blocking parsers (pdfplumber,
python-docx) run in a worker thread so the async ingestion pipeline never blocks
the event loop; the web loader is natively async.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load a PDF into one content block per (non-empty) page."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)

    def _load_sync(self) -> list[dict[str, Any]]:
        import pdfplumber

        source = self.filepath.name
        blocks: list[dict[str, Any]] = []
        with pdfplumber.open(str(self.filepath)) as pdf:
            # Cap page count before extracting any text (P0.4) so a huge PDF
            # can't pin CPU/memory in the parse loop.
            if settings.max_pdf_pages and len(pdf.pages) > settings.max_pdf_pages:
                raise ValueError(
                    f"PDF has {len(pdf.pages)} pages (max {settings.max_pdf_pages})"
                )
            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                blocks.append(
                    {"content": text, "page_number": page_number, "source": source}
                )
        logger.info("PDFLoader: %s -> %d page blocks", source, len(blocks))
        return blocks

    async def load(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._load_sync)


class DocxLoader:
    """Load a .docx into one content block per heading-delimited section."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)

    def _load_sync(self) -> list[dict[str, Any]]:
        from docx import Document as DocxDocument

        source = self.filepath.name
        doc = DocxDocument(str(self.filepath))
        blocks: list[dict[str, Any]] = []
        current_section = "Introduction"
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(
                    {"content": text, "section": current_section, "source": source}
                )
            buffer = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "") if para.style else ""
            if style.startswith("Heading") or style == "Title":
                flush()
                current_section = text
            else:
                buffer.append(text)
        flush()
        logger.info("DocxLoader: %s -> %d section blocks", source, len(blocks))
        return blocks

    async def load(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._load_sync)


class WebLoader:
    """Load a web page's main text. Prefers Tavily extract (if a key is set),
    falling back to httpx + BeautifulSoup."""

    def __init__(self, url: str) -> None:
        self.url = url

    async def _load_tavily(self) -> list[dict[str, Any]] | None:
        if not settings.tavily_api_key:
            return None
        try:
            from tavily import TavilyClient

            def _extract() -> dict[str, Any]:
                client = TavilyClient(api_key=settings.tavily_api_key)
                return client.extract(urls=[self.url])

            res = await asyncio.to_thread(_extract)
            results = (res or {}).get("results") or []
            blocks = [
                {"content": (r.get("raw_content") or "").strip(),
                 "url": r.get("url", self.url),
                 "title": r.get("url", self.url)}
                for r in results
                if (r.get("raw_content") or "").strip()
            ]
            return blocks or None
        except Exception as exc:  # noqa: BLE001 -- fall back to httpx on any Tavily error
            logger.warning("WebLoader: Tavily extract failed (%s); using httpx", exc)
            return None

    async def _load_httpx(self) -> list[dict[str, Any]]:
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=20.0,
            headers={"User-Agent": "BriefrBot/1.0"},
        ) as client:
            resp = await client.get(self.url)
            resp.raise_for_status()
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else self.url
        )
        text = "\n".join(
            line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
        )
        return [{"content": text, "url": self.url, "title": title}] if text else []

    async def load(self) -> list[dict[str, Any]]:
        blocks = await self._load_tavily()
        if blocks is None:
            blocks = await self._load_httpx()
        logger.info("WebLoader: %s -> %d blocks", self.url, len(blocks))
        return blocks
