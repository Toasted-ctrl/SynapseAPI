from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.admin_users import (
    NoAdminError,
    UserAdminSettings,
    get_admin_user_by_user_id,
)
from database.person import PersonDetails, get_person_by_person_id
from database.session import get_db_session
from database.user import UserDetails, get_user_by_user_id

from .models import AuthenticatedUserRead

log = get_logger()


router = APIRouter()


@router.get(
    "/user/me",
    tags=["User"],
    response_model=AuthenticatedUserRead,
    description="Returns the user_id, first name, last name and email address of the authenticated User.",
)
async def me(
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> AuthenticatedUserRead:

    u: UserDetails | None = await get_user_by_user_id(session=session, user_id=user.id)
    if u is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )

    p: PersonDetails | None = await get_person_by_person_id(
        session=session, person_id=u.person_id
    )
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Person not found"
        )

    a: UserAdminSettings | None = None
    try:
        a = await get_admin_user_by_user_id(user_id=user.id, session=session)
    except NoAdminError as e:
        log.debug(str(e))

    return AuthenticatedUserRead(
        user_id=u.user_id,
        first_name=p.first_name,
        last_name=p.last_name,
        email=p.email,
        is_admin=True if a else False,
        is_super_admin=True if a and a.is_super_admin else False,
    )
