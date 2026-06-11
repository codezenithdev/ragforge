"""ORM models package.

Importing this package registers every model on ``Base.metadata`` (used by
``init_models()`` and Alembic).
"""

from app.models.brief import Brief, BriefStatus, BriefSubQuery
from app.models.document import Document, DocumentStatus, SourceType

__all__ = [
    "Document",
    "DocumentStatus",
    "SourceType",
    "Brief",
    "BriefStatus",
    "BriefSubQuery",
]
