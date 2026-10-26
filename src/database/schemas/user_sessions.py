import uuid
from datetime import datetime

from sqlalchemy import UUID, DateTime, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class SessionsT(Base):
    __tablename__ = "user_sessions"

    session_id_hash: Mapped[str] = mapped_column(String(255), primary_key=True)

    user_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    expiration_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now() + text("INTERVAL '24 hours'"),
    )
