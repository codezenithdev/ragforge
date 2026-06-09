"""ORM models package.

Importing this package registers every model on ``Base.metadata`` (used by
``init_models()`` and Alembic).
"""

from app.models.brief import Brief, BriefStatus, BriefSubQuery
from app.models.document import Document, SourceType

__all__ = [
    "Document",
    "SourceType",
    "Brief",
    "BriefStatus",
    "BriefSubQuery",
]
