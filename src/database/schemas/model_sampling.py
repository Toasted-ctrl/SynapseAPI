from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from database.schemas.base import Base


class ModelSamplingT(Base):
    __tablename__ = "model_sampling"

    name: Mapped[str] = mapped_column(String(255), primary_key=True)

    provider: Mapped[str] = mapped_column(String(255), primary_key=True)

    parameter: Mapped[str] = mapped_column(String(255), primary_key=True)

    supported: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    created_by: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=text("current_user")
    )
