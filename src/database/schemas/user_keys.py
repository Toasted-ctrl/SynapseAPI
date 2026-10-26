import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import UUID, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class UserKeysT(Base):
    __tablename__ = "user_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    user_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=False, unique=False)

    provider_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=False, unique=False)

    encrypted_api_key: Mapped[str] = mapped_column(
        String(500), nullable=False, unique=True
    )

    api_key_short: Mapped[str] = mapped_column(String(10), nullable=False, unique=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, server_default=func.now()
    )

    expiration_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        default=lambda: datetime.now(UTC).replace(tzinfo=None) + timedelta(days=30),
    )
