from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.config import config
from core.logging import get_logger
from database.providers import ProviderConfiguration, get_all_provider_configurations
from database.session import get_db_session
from providers.vector_embedding import get_embedding

from .models import VectorEmbeddingCreate, VectorEmbeddingRead

router = APIRouter()


tags = ["Vector Store / Embedding"]


log = get_logger()


@router.post(
    path="/vector_embedding",
    tags=tags,
    description=(
        "Returns a vector embedding of the query for the indicated Provider and model."
    ),
    response_model=VectorEmbeddingRead,
)
async def post_test_vector_embedding(
    payload: VectorEmbeddingCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> VectorEmbeddingRead:

    try:
        if payload.model not in config.VECTOR_EMBEDDING_MODELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{payload.model}' is not a valid embedding model",
            )

        p_reg = await get_all_provider_configurations(session=session, user_id=user.id)

        if (
            payload.provider not in p_reg.names
            or payload.provider in p_reg.not_configured
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{payload.provider}' is not supported or not configured",
            )

        prov: ProviderConfiguration = getattr(p_reg, payload.provider)

        embedding = await get_embedding(
            langchain_con=prov.langchain_con,
            model=payload.model,
            base_url=prov.base_url,
            prompt=payload.prompt,
            dimensions=payload.dimensions,
            encrypted_api_key=prov.encrypted_api_key,
        )

        return VectorEmbeddingRead(
            prompt=payload.prompt,
            provider=payload.provider,
            model=payload.model,
            dimensions=len(embedding),
            embedding=embedding,
        )

    except ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to connect to Provider '{payload.provider}'",
        )
