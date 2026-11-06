import uuid

import pytest
from db_helpers import make_session, params

from database.admin_users import NoAdminError, get_admin_user_by_user_id

pytestmark = pytest.mark.asyncio


class TestGetAdminUserByUserId:
    """Tests for the get_admin_user_by_user_id database function."""

    async def test_returns_user_id_when_user_is_admin(self):
        """Test that user_id is returned when user is an admin."""
        user_id = uuid.uuid4()
        # The query selects only AdminT.user_id, so scalar_one_or_none returns just the UUID
        session = make_session(execute=[user_id])

        result = await get_admin_user_by_user_id(user_id=user_id, session=session)

        assert result == user_id

    async def test_raises_no_admin_error_when_user_is_not_admin(self):
        """Test that NoAdminError is raised when user is not an admin."""
        user_id = uuid.uuid4()
        session = make_session(execute=[None])

        with pytest.raises(NoAdminError) as exc_info:
            await get_admin_user_by_user_id(user_id=user_id, session=session)

        assert f"User {user_id} is not an Admin." in str(exc_info.value)

    async def test_filters_on_user_id(self):
        """Test that the query filters on user_id."""
        user_id = uuid.uuid4()
        session = make_session(execute=[user_id])

        await get_admin_user_by_user_id(user_id=user_id, session=session)

        query_params = params(session.execute.await_args)
        assert user_id in query_params

    async def test_returns_scalar_value(self):
        """Test that the function returns a scalar UUID value, not a row."""
        user_id = uuid.uuid4()
        session = make_session(execute=[user_id])

        result = await get_admin_user_by_user_id(user_id=user_id, session=session)

        assert isinstance(result, uuid.UUID)
        assert result == user_id
