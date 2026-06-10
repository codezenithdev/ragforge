"""Celery application.

Redis is both broker and result backend. Tasks live in ``app.tasks``; the brief
pipeline is async, so the task module maintains a persistent per-worker event
loop (see ``app.tasks``) rather than spinning a new loop per task — the cached
async clients (Chroma, Anthropic, asyncpg pool) are loop-bound and must not
outlive their loop.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "briefr",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    # One long-running LLM pipeline at a time per worker process.
    worker_prefetch_multiplier=1,
    task_acks_late=False,
)
