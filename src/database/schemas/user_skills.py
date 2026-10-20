import uuid
from datetime import datetime
from sqlalchemy import UUID, String, DateTime, func, text, JSON, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class UserSkillsT(Base):
    __tablename__ = "user_skills"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        nullable=False,
        index=True
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    description: Mapped[str] = mapped_column(
        String(1000),
        nullable=True
    )

    skill_text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    parameters_schema: Mapped[dict] = mapped_column(
        JSON,
        nullable=True
    )

    is_encrypted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    created_by: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default=text("current_user")
    )