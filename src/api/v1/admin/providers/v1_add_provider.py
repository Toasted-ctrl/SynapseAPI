from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import depends_verify_admin
from database.providers import ProviderExistsError, insert_provider_record
from database.session import get_db_session

from .models import ProviderCreate, ProviderRead

router = APIRouter()


@router.post(
    "/admin/providers",
    tags=["Admin / Providers"],
    description="Register a new provider. Requires Admin permissions.",
    response_model=ProviderRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(depends_verify_admin)],
)
async def admin_add_provider(
    provider: ProviderCreate,
    session: AsyncSession = Depends(get_db_session),
) -> ProviderRead:
    """Add a new provider to the database.

    This endpoint creates a new provider record in the database. It checks for
    duplicate providers based on the combination of name, base_url, internal,
    requires_api_key, langchain_con, and mac_address.

    Args:
        provider: The provider data to create.
        session: The async SQLAlchemy session.

    Returns:
        ProviderRead: The newly created provider record.

    Raises:
        HTTPException 409: If a provider with the same attributes already exists.
    """
    try:
        created = await insert_provider_record(
            name=provider.name,
            base_url=provider.base_url,
            internal=provider.internal,
            requires_api_key=provider.requires_api_key,
            langchain_con=provider.langchain_con,
            session=session,
            mac_address=provider.mac_address,
        )
    except ProviderExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return ProviderRead(**asdict(created))
