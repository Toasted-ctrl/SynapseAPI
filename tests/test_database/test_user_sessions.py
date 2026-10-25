import uuid

import pytest
from db_helpers import make_session

from database.schemas.user_sessions import SessionsT
from database.user_sessions import post_session
from security.hash import get_hash_sha256

pytestmark = pytest.mark.asyncio


class TestPostSession:
    async def test_stores_hash_and_returns_raw_session_id(self):
        user_id = uuid.uuid4()
        session = make_session()

        session_id = await post_session(session=session, user_id=user_id)

        [added] = session.added
        assert isinstance(added, SessionsT)
        assert added.user_id == user_id
        assert added.session_id_hash == get_hash_sha256(session_id)
        assert added.session_id_hash != session_id

    async def test_generates_unique_session_ids(self):
        session = make_session()
        user_id = uuid.uuid4()

        first = await post_session(session=session, user_id=user_id)
        second = await post_session(session=session, user_id=user_id)

        assert first != second
