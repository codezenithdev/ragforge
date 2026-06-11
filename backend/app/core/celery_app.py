"""Celery application.

Redis is both broker and result backend. Tasks live in ``app.tasks``; the brief
pipeline is async, so the task module maintains a persistent per-worker event
loop (see ``app.tasks``) rather than spinning a new loop per task — the cached
async clients (Chroma, Anthropic, asyncpg pool) are loop-bound and must not
outlive their loop.
"""

from __future__ import annotations

from celery import Celery
from celery.signals import setup_logging

from app.core.config import settings


@setup_logging.connect
def _configure_worker_logging(**_kwargs: object) -> None:
    """Use the same structured JSON logging in the worker as the API (P2.2)."""
    from app.core.logging import configure_logging

    configure_logging()

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
    # P1.2: ack only after the task finishes so a worker crash redelivers the job
    # (idempotency in _run_pipeline makes re-delivery safe); a lost worker requeues.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Periodic maintenance (run beat alongside the worker: `-B` in dev; a dedicated
    # beat service in prod).
    beat_schedule={
        # P1.4: purge orphaned Chroma vectors and fail documents stuck mid-ingest.
        "reconcile-storage": {
            "task": "app.tasks.reconcile_storage_task",
            "schedule": float(settings.reconcile_interval_seconds),
        },
        # P1.2: fail briefs left in 'processing' past the deadline (worker lost).
        "sweep-stuck-briefs": {
            "task": "app.tasks.sweep_stuck_briefs_task",
            "schedule": float(settings.brief_sweep_interval_seconds),
        },
        # P2.3: prune briefs past the retention window (no-op unless configured).
        "prune-old-briefs": {
            "task": "app.tasks.prune_old_briefs_task",
            "schedule": 86400.0,
        },
    },
)
