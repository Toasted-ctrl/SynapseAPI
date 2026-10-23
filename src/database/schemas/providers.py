from sqlalchemy import (
    String,
    Boolean,
    UUID
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column
)
import uuid

from database.schemas.base import Base


class ProvidersT(Base):
    __tablename__ = "providers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=lambda: uuid.uuid4()
    )

    name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True
    )

    base_url: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True
    )

    langchain_con: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=False
    )

    host: Mapped[str] = mapped_column(
        String(30),
        nullable=True,
        unique=False
    )

    mac_address: Mapped[str] = mapped_column(
        String(30),
        nullable=True,
        unique=True
    )

    internal: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False
    )

    requires_api_key: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False
    )