import uuid
from datetime import datetime

from sqlalchemy import (
    UUID,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class AgentConfigT(Base):
    __tablename__ = "agent_configs"
    __table_args__ = (
        UniqueConstraint("owner_user_id", "name", name="uq_agent_configs_owner_name"),
        CheckConstraint(
            "agent_type IN ('Conversational', 'Decision')",
            name="ck_agent_configs_agent_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    settings: Mapped[dict] = mapped_column(JSONB, nullable=False)

    agent_type: Mapped[str] = mapped_column(String(255), nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, server_default=func.now()
    )

    updated_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
