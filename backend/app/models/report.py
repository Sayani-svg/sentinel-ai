"""Report ORM model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Report(Base):
    """Persisted generated security report."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    report_path: Mapped[str]
    generated_at: Mapped[datetime]
