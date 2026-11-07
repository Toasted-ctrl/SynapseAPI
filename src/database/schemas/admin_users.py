import uuid
from datetime import datetime

from sqlalchemy import UUID, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class AdminT(Base):
    __tablename__ = "admin_users"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )

    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False)

    is_super_admin: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, server_default=func.now()
    )

    updated_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
