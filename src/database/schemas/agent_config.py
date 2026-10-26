import uuid
from datetime import datetime

from sqlalchemy import UUID, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class AgentConfigT(Base):
    __tablename__ = "agent_configs"
    __table_args__ = (
        UniqueConstraint("owner_user_id", "name", name="uq_agent_configs_owner_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    settings: Mapped[dict] = mapped_column(JSONB, nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, server_default=func.now()
    )

    updated_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
