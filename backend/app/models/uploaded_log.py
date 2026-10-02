"""UploadedLog ORM model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class UploadedLog(Base):
    """Persisted user-uploaded log file."""

    __tablename__ = "uploaded_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str]
    file_path: Mapped[str]
    upload_status: Mapped[str]
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime]
