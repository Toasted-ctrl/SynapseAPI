from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.providers import (
    ProviderConfiguration,
    UserProviderRegistry,
    get_all_provider_configurations,
)
from database.session import get_db_session
from providers.vector_embedding import get_embedding

from .helpers import ModelType, VectorEmbeddingSettings, get_provider_configuration
from .models import VectorEmbeddingCreate, VectorEmbeddingRead

router = APIRouter(tags=["Inference / Vector Embedding"])


log = get_logger()


@router.post(
    path="/inference/vector-embedding",
    description="Generates a vector embedding of the prompt with the requested embedding model.",
    response_model=VectorEmbeddingRead,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Provider is not configured or does not support embeddings."
        },
        status.HTTP_404_NOT_FOUND: {"description": "Provider or model does not exist."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Unable to connect to provider."
        },
    },
)
async def get_vector_embedding(
    payload: VectorEmbeddingCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> VectorEmbeddingRead:
    upr: UserProviderRegistry = await get_all_provider_configurations(
        session=session, user_id=user.id
    )

    prov: ProviderConfiguration = await get_provider_configuration(
        user_provider_registry=upr,
        settings=VectorEmbeddingSettings(model=payload.model),
        model_type=ModelType.VECTOR_EMBEDDING,
    )

    try:
        embedding = await get_embedding(
            langchain_con=prov.langchain_con,
            model=payload.model.name,
            base_url=prov.base_url,
            prompt=payload.prompt,
            dimensions=payload.dimensions,
            encrypted_api_key=prov.encrypted_api_key,
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider does not support embeddings: {prov.name}",
        )
    except ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )

    return VectorEmbeddingRead(
        model=payload.model,
        prompt=payload.prompt,
        dimensions=len(embedding),
        embedding=embedding,
    )
