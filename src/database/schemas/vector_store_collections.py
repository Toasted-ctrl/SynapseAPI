import uuid

from sqlalchemy import ARRAY, UUID, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class VectorStoreCollectionT(Base):
    __tablename__ = "vector_store_collections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    vector_store_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=False)

    e_provider: Mapped[str] = mapped_column(String(255), nullable=False)

    e_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)

    e_model: Mapped[str] = mapped_column(String(50), nullable=False)

    e_api_key: Mapped[str] = mapped_column(String(255), nullable=False)

    scope: Mapped[str] = mapped_column(String(30), nullable=False)

    # TODO: Probably not required filters, but rather required metadata.
    required_filters: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
