import uuid

import pytest
from db_helpers import make_session, params
from fastapi import HTTPException

from database.message_threads import store_thread_id, verify_or_get_thread_id
from database.schemas.message_threads import MessageThreadsT

pytestmark = pytest.mark.asyncio


class TestStoreThreadId:
    async def test_stores_and_returns_thread_id(self):
        thread_id, user_id = uuid.uuid4(), uuid.uuid4()
        session = make_session()

        result = await store_thread_id(
            session=session, thread_id=thread_id, user_id=user_id
        )

        [added] = session.added
        assert isinstance(added, MessageThreadsT)
        assert added.id == thread_id
        assert added.user_id == user_id
        session.flush.assert_awaited_once()
        assert result == thread_id


class TestVerifyOrGetThreadId:
    async def test_returns_thread_id_owned_by_user(self):
        thread_id, user_id = uuid.uuid4(), uuid.uuid4()
        session = make_session(scalar=[thread_id])

        result = await verify_or_get_thread_id(
            session=session, user_id=user_id, thread_id=thread_id
        )

        assert result == thread_id
        assert params(session.scalar.await_args) == [user_id, thread_id]
        session.add.assert_not_called()

    async def test_raises_404_for_thread_not_owned_by_user(self):
        session = make_session(scalar=[None])

        with pytest.raises(HTTPException) as exc:
            await verify_or_get_thread_id(
                session=session, user_id=uuid.uuid4(), thread_id=uuid.uuid4()
            )

        assert exc.value.status_code == 404

    async def test_creates_new_thread_when_none_given(self):
        user_id = uuid.uuid4()
        session = make_session()

        result = await verify_or_get_thread_id(session=session, user_id=user_id)

        [added] = session.added
        assert isinstance(result, uuid.UUID)
        assert added.id == result
        assert added.user_id == user_id
        session.scalar.assert_not_awaited()
