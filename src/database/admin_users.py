from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

from .schemas.admin_users import AdminT

log = get_logger()


class NoAdminError(ValueError):
    """Raised when the User is not an Admin."""

    pass


@dataclass(frozen=True)
class UserAdminSettings:
    user_id: UUID
    is_super_admin: bool


async def get_admin_user_by_user_id(
    user_id: UUID, session: AsyncSession
) -> UserAdminSettings:
    """Returns the UserAdminSettings if the User is registered as Admin.

    Raises:
        NoAdminError: If the User is not registered as Admin.
    """
    admin = (
        await session.execute(select(AdminT).where(AdminT.user_id == user_id))
    ).scalar_one_or_none()
    if not admin:
        raise NoAdminError(f"User {user_id} is not an Admin.")
    return UserAdminSettings(user_id=admin.user_id, is_super_admin=admin.is_super_admin)


async def add_user_as_admin(user_id: UUID, session: AsyncSession) -> UserAdminSettings:
    """Adds a User as a basic Admin to the AdminT table. This will enable basic Administrator access."""
    try:
        admin = await get_admin_user_by_user_id(user_id=user_id, session=session)
        if admin:
            log.debug(f"User {user_id} is already an Admin.")
        return admin
    except NoAdminError:
        log.debug(f"Adding User {user_id} as Admin ...")
        new = AdminT(user_id=user_id, is_super_admin=False)
        session.add(new)
        await session.flush()
        return UserAdminSettings(user_id=user_id, is_super_admin=False)
