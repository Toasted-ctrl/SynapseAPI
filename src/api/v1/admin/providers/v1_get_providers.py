from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import depends_verify_admin
from database.providers import NoProvidersFoundError, get_all_provider_records
from database.session import get_db_session

from .models import ProviderListRead, ProviderRead

router = APIRouter()


@router.get(
    "/admin/providers",
    tags=["Admin / Providers"],
    description="Get all providers. Requires Admin permissions.",
    response_model=ProviderListRead,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No providers found."},
    },
    dependencies=[Depends(depends_verify_admin)],
)
async def admin_get_providers(
    session: AsyncSession = Depends(get_db_session),
) -> ProviderListRead:
    """Retrieve all provider records from the database.

    Args:
        session: The async SQLAlchemy session.

    Returns:
        ProviderListRead: A list of all provider records.

    Raises:
        HTTPException 404: If no providers are found in the database.
    """
    try:
        providers = await get_all_provider_records(session=session)
    except NoProvidersFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No providers found in the database.",
        )
    return ProviderListRead(
        providers=[ProviderRead(**asdict(p)) for p in providers]
    )
