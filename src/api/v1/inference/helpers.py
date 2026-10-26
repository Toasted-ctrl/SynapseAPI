from dataclasses import dataclass
from enum import StrEnum

from fastapi import HTTPException, status

from api.v1.playground.models import AgentModel, AgentTools
from core.model_types import model_config
from database.providers import ProviderConfiguration, UserProviderRegistry


@dataclass(frozen=True)
class ConversationSettings:
    model: AgentModel
    system_prompt: str | None
    tools: AgentTools


class ModelType(StrEnum):
    CHAT_COMPLETION = "chat_completion"
    VECTOR_EMBEDDING = "vector_embedding"
    TRANSLATION = "translation"


async def _validate_supported_model(model_name: str, model_type: ModelType) -> None:
    match model_type:
        case ModelType.CHAT_COMPLETION:
            supported = await model_config.chat_completion_models()
        case ModelType.TRANSLATION:
            supported = await model_config.translation_models()
        case ModelType.VECTOR_EMBEDDING:
            supported = await model_config.vector_embedding_models()
    if model_name not in supported:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unsupported {model_type} model: {model_name}",
        )


async def get_provider_configuration(
    user_provider_registry: UserProviderRegistry,
    settings: ConversationSettings,
    model_type: ModelType,
) -> ProviderConfiguration:
    """Retrieves the ProviderConfiguration required for the conversation request.
    Raises an error if not configured, or the Provider does not support the model."""
    if settings.model.provider not in user_provider_registry.names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unsupported provider: {settings.model.provider}",
        )
    if settings.model.provider in user_provider_registry.not_configured:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider not configured: {settings.model.provider}",
        )
    await _validate_supported_model(
        model_name=settings.model.name, model_type=model_type
    )
    return user_provider_registry[settings.model.provider]
