from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas.admin_users import AdminT


class NoAdminError(ValueError):
    """Raised when the User is not an Admin."""

    pass


async def get_admin_user_by_user_id(user_id: UUID, session: AsyncSession) -> UUID:
    """Returns the user_id if the User is registered as Admin."""
    user = (
        await session.execute(
            select(AdminT.user_id).where(AdminT.user_id == user_id, AdminT.is_admin)
        )
    ).scalar_one_or_none()
    if not user:
        raise NoAdminError(f"User {user_id} is not an Admin.")
    return user
