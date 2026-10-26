import uuid
from datetime import datetime

from sqlalchemy import UUID, DateTime, ForeignKey, String, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.schemas.base import Base


class PersonsT(Base):
    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    encrypted_email: Mapped[str] = mapped_column(String(255), nullable=False)

    encrypted_first_name: Mapped[str] = mapped_column(String(255), nullable=False)

    encrypted_last_name: Mapped[str] = mapped_column(String(255), nullable=False)

    blind_index_email: Mapped[str] = mapped_column(String(255), nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    created_by: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=text("current_user")
    )

    # Relationships
    memberships: Mapped[list[UsersT]] = relationship(back_populates="person")


class UsersT(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=lambda: uuid.uuid4()
    )

    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("persons.id"), nullable=False
    )

    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("clients.id"), nullable=False
    )

    external_id: Mapped[str] = mapped_column(
        # This is where we will map the external id (openid sub) to.
        String(255),
        nullable=True,
    )

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    login_provider: Mapped[str] = mapped_column(
        # Required for frontend applications that support external login providers.
        String(50),
        nullable=True,
    )

    created_by: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=text("current_user")
    )

    # Relationships
    person: Mapped[PersonsT] = relationship(back_populates="memberships")
