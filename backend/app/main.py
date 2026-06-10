"""Briefr API entrypoint.

FastAPI app with CORS, an async lifespan that prepares Postgres tables and
verifies the ChromaDB server is reachable, and the v1 routers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.models  # noqa: F401  -- register ORM models on Base.metadata
from app.api.routes import briefs, documents, eval as eval_routes
from app.core.config import settings
from app.core.database import init_models
from app.core.vector_db import chroma_heartbeat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await init_models()
    await chroma_heartbeat()
    logger.info("Briefr API ready (environment=%s)", settings.environment)
    yield


app = FastAPI(title="Briefr", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1")
app.include_router(briefs.router, prefix="/api/v1")
app.include_router(eval_routes.router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "environment": settings.environment}
