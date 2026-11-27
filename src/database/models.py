from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.schemas.models import ModelsT


class ModelAlreadyExistsError(ValueError):
    """Raised when a Model with the same name is already registered."""

    pass


class ModelNotFoundError(ValueError):
    """Raised when a Model with the given name is not registered."""

    pass


async def get_models_by_expertise(session: AsyncSession, expertise: str) -> list[str]:

    models = (
        await session.scalars(select(ModelsT).where(ModelsT.expertise == expertise))
    ).all()

    return [m.name for m in models]


@dataclass(frozen=True)
class ModelWithExpertise:
    """Represents a model with its associated expertise.

    Attributes:
        model_name: The name of the model.
        expertise: The expertise category of the model (e.g., chat_completion, translation).
        created_date: The datetime when the model was created.
    """

    model_name: str
    expertise: str
    created_date: datetime


async def add_model_with_expertise(
    session: AsyncSession, model_name: str, expertise: str
) -> ModelWithExpertise:
    """Add a new model with a specific expertise to the database.

    Args:
        session: The async database session.
        model_name: The name of the model to add.
        expertise: The expertise category for the model.

    Returns:
        ModelWithExpertise: The created model with its metadata.

    Raises:
        ModelAlreadyExistsError: If a model with the same name already exists.
            The model name is the primary key, so a name can only have one expertise.
    """
    existing = (
        await session.execute(select(ModelsT).where(ModelsT.name == model_name))
    ).scalar_one_or_none()
    if existing:
        raise ModelAlreadyExistsError(
            f"Model {model_name} already exists with expertise {existing.expertise}."
        )
    model = ModelsT(name=model_name, expertise=expertise)
    session.add(model)
    await session.flush()
    await session.refresh(model)
    return ModelWithExpertise(
        model_name=model.name,
        expertise=model.expertise,
        created_date=model.created_date,
    )


async def delete_model_by_name(
    session: AsyncSession, model_name: str
) -> ModelWithExpertise:
    """Delete a model from the database.

    Args:
        session: The async database session.
        model_name: The name of the model to delete.

    Returns:
        ModelWithExpertise: The deleted model, so callers know which expertise it had.

    Raises:
        ModelNotFoundError: If no model with that name exists.
    """
    model = (
        await session.execute(select(ModelsT).where(ModelsT.name == model_name))
    ).scalar_one_or_none()
    if not model:
        raise ModelNotFoundError(f"Model {model_name} does not exist.")
    await session.delete(model)
    return ModelWithExpertise(
        model_name=model.name,
        expertise=model.expertise,
        created_date=model.created_date,
    )
