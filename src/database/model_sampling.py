from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas.model_sampling import ModelSamplingT


@dataclass(frozen=True)
class ModelSampling:
    temperature: bool = False
    top_p: bool = False
    top_k: bool = False


async def get_model_sampling_settings(
    session: AsyncSession, provider_name: str, model_name: str
) -> ModelSampling:
    """Will check available model sampling settings for the indicated model and provider.
    Returns a ModelSampling object."""
    sampling = (
        await session.scalars(
            select(ModelSamplingT).where(
                ModelSamplingT.provider == provider_name,
                ModelSamplingT.name == model_name,
            )
        )
    ).all()
    sampling_dict = {s.parameter: s.supported for s in sampling}
    return ModelSampling(
        temperature=sampling_dict.get("temperature", False),
        top_k=sampling_dict.get("top_k", False),
        top_p=sampling_dict.get("top_p", False),
    )
