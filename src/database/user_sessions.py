import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from security.hash import get_hash_sha256
from security.secret import create_secret

from .schemas.user_sessions import SessionsT

log = get_logger()


async def post_session(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> str:
    """Creates a new user session in the database and returns a session_id for the cookie."""
    log.debug(f"Creating new User session for User'{user_id}' ...")
    session_id = create_secret()
    ns = SessionsT(session_id_hash=get_hash_sha256(session_id), user_id=user_id)
    session.add(ns)
    log.debug(f"New session created for User '{user_id}'.")
    return session_id
