"""Async relational database layer (Postgres).

Holds the SQLAlchemy async engine + session factory used by the ORM models
(documents, briefs, brief_sub_queries) and the FastAPI request lifecycle.

Vectors live in ChromaDB, not Postgres — see ``app.core.vector_db``. This module
deliberately knows nothing about embeddings.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.core.config import settings

logger = logging.getLogger(__name__)

# Declarative base shared by every ORM model (Phase 2+).
Base = declarative_base()

# Async SQLAlchemy engine + session factory.
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yield an async session, closing it afterwards."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_models() -> None:
    """Create all tables registered on ``Base``. Called from the app lifespan."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
