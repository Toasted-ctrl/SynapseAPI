from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.schemas.models import ModelsT


async def get_models_by_expertise(session: AsyncSession, expertise: str) -> list[str]:

    models = (
        await session.scalars(select(ModelsT).where(ModelsT.expertise == expertise))
    ).all()

    return [m.name for m in models]
