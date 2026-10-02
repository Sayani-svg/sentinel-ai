"""ModelVersion ORM model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ModelVersion(Base):
    """Persisted machine learning model version."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    version_tag: Mapped[str]
    model_type: Mapped[str]
    accuracy: Mapped[float]
    precision: Mapped[float]
    recall: Mapped[float]
    f1_score: Mapped[float]
    is_active: Mapped[bool]
    created_at: Mapped[datetime]
