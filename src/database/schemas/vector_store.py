import uuid

from sqlalchemy import UUID, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class VectorStoreSettingsT(Base):
    __tablename__ = "vector_store_settings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    encrypted_api_key: Mapped[str] = mapped_column(
        String(500), nullable=True, unique=True
    )

    vendor: Mapped[str] = mapped_column(String(50), nullable=False)

    base_url: Mapped[str] = mapped_column(String(255), nullable=False)

    port: Mapped[int] = mapped_column(Integer, nullable=True)
