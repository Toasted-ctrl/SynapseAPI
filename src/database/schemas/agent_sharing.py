import uuid
from datetime import datetime

from sqlalchemy import UUID, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class AgentSharingT(Base):
    __tablename__ = "agent_sharing"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("agent_configs.id", ondelete="CASCADE"), primary_key=True
    )

    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("users.id", ondelete="CASCADE")
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, server_default=func.now()
    )
