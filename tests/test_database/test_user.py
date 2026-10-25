import uuid

import pytest
from db_helpers import make_session, params, row

from database.schemas.persons_users import UsersT
from database.user import User, UserDetails, get_or_store_user, get_user_by_user_id

pytestmark = pytest.mark.asyncio


class TestGetUserByUserId:
    async def test_returns_user_details(self):
        user_id, person_id = uuid.uuid4(), uuid.uuid4()
        session = make_session(execute=[row(id=user_id, person_id=person_id)])

        result = await get_user_by_user_id(session=session, user_id=user_id)

        assert result == UserDetails(user_id=user_id, person_id=person_id)

    async def test_returns_none_when_not_found(self):
        session = make_session(execute=[None])

        assert await get_user_by_user_id(session=session, user_id=uuid.uuid4()) is None

    async def test_filters_on_user_id(self):
        user_id = uuid.uuid4()
        session = make_session(execute=[None])

        await get_user_by_user_id(session=session, user_id=user_id)

        assert params(session.execute.await_args) == [user_id]


class TestGetOrStoreUser:
    async def test_returns_existing_user_without_insert(self):
        existing_id = uuid.uuid4()
        session = make_session(scalar=[row(id=existing_id)])

        result = await get_or_store_user(
            session=session,
            person_id=uuid.uuid4(),
            api_key_id=uuid.uuid4(),
            key_type="User",
        )

        assert result == User(id=existing_id)
        session.add.assert_not_called()
        session.flush.assert_not_awaited()

    async def test_creates_user_when_missing(self):
        person_id, api_key_id = uuid.uuid4(), uuid.uuid4()
        session = make_session(scalar=[None])

        result = await get_or_store_user(
            session=session,
            person_id=person_id,
            api_key_id=api_key_id,
            key_type="Application",
            login_provider="Google",
            external_id="google-sub",
        )

        [added] = session.added
        assert isinstance(added, UsersT)
        assert added.person_id == person_id
        assert added.api_key_id == api_key_id
        assert added.login_provider == "Google"
        assert added.external_id == "google-sub"
        session.flush.assert_awaited_once()
        assert result == User(id=added.id)

    async def test_lookup_filters_on_all_identifying_fields(self):
        person_id, api_key_id = uuid.uuid4(), uuid.uuid4()
        session = make_session(scalar=[row(id=uuid.uuid4())])

        await get_or_store_user(
            session=session,
            person_id=person_id,
            api_key_id=api_key_id,
            key_type="Application",
            login_provider="Google",
            external_id="google-sub",
        )

        assert {person_id, api_key_id, "Google", "google-sub"} <= set(
            params(session.scalar.await_args)
        )

    async def test_application_user_requires_external_id(self):
        session = make_session()

        with pytest.raises(ValueError, match="external_id must not be None"):
            await get_or_store_user(
                session=session,
                person_id=uuid.uuid4(),
                api_key_id=uuid.uuid4(),
                key_type="Application",
            )

        session.scalar.assert_not_awaited()
