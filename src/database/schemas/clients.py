import uuid
from datetime import datetime

from sqlalchemy import (
    UUID,
    Boolean,
    CheckConstraint,
    DateTime,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class ClientsT(Base):
    __tablename__ = "clients"

    __table_args__ = (
        CheckConstraint("key_type IN ('User', 'Application')", name="check_key_type"),
        UniqueConstraint(
            "blind_index_client_name",
            "blind_index_owner_email",
            name="uq_client_owner_email",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    api_key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    key_type: Mapped[str] = mapped_column(String(50), nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    require_jwt: Mapped[bool] = mapped_column(Boolean, nullable=False)

    require_external_id: Mapped[bool] = mapped_column(Boolean, nullable=False)

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )

    encrypted_owner_email: Mapped[str] = mapped_column(String(255), nullable=False)

    encrypted_hmac_secret: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )

    encrypted_client_name: Mapped[str] = mapped_column(String(100), nullable=False)

    encrypted_redirect_uri: Mapped[str] = mapped_column(
        String(255), nullable=True, unique=True
    )

    blind_index_client_name: Mapped[str] = mapped_column(String(255), nullable=False)

    blind_index_owner_email: Mapped[str] = mapped_column(String(255), nullable=False)

    created_by: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=text("current_user")
    )
