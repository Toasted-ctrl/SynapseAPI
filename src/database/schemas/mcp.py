import uuid
from datetime import datetime

from sqlalchemy import UUID, DateTime, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class MCPsT(Base):
    __tablename__ = "mcps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    url: Mapped[str] = mapped_column(String(255), nullable=False)

    transport: Mapped[str] = mapped_column(String(20), nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    created_by: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=text("current_user")
    )
