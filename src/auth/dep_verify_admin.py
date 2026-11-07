from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.admin_users import NoAdminError, get_admin_user_by_user_id
from database.session import get_db_session

from .dep_verify_user import VerifiedUser, depends_verify_user


@dataclass(frozen=True)
class VerifiedAdmin:
    id: UUID


async def depends_verify_admin(
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> VerifiedAdmin:
    try:
        user_id = await get_admin_user_by_user_id(user_id=user.id, session=session)
    except NoAdminError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e)
    return VerifiedAdmin(id=user_id)
