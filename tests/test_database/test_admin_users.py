import uuid

import pytest
from db_helpers import make_session, params, row

from database.admin_users import (
    NoAdminError,
    UserAdminSettings,
    add_user_as_admin,
    get_admin_user_by_user_id,
)
from database.schemas.admin_users import AdminT

pytestmark = pytest.mark.asyncio


def make_admin(user_id: uuid.UUID, is_super_admin: bool = False):
    """A fake AdminT row as returned by the mocked session."""
    return row(user_id=user_id, is_super_admin=is_super_admin)


class TestGetAdminUserByUserId:
    """Tests for the get_admin_user_by_user_id database function."""

    async def test_returns_admin_settings_when_user_is_admin(self):
        """Test that UserAdminSettings is returned when user is an admin."""
        user_id = uuid.uuid4()
        session = make_session(execute=[make_admin(user_id)])

        result = await get_admin_user_by_user_id(user_id=user_id, session=session)

        assert isinstance(result, UserAdminSettings)
        assert result.user_id == user_id
        assert result.is_super_admin is False

    async def test_returns_super_admin_flag(self):
        """Test that the is_super_admin flag is taken from the admin row."""
        user_id = uuid.uuid4()
        session = make_session(execute=[make_admin(user_id, is_super_admin=True)])

        result = await get_admin_user_by_user_id(user_id=user_id, session=session)

        assert result.is_super_admin is True

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
        session = make_session(execute=[make_admin(user_id)])

        await get_admin_user_by_user_id(user_id=user_id, session=session)

        query_params = params(session.execute.await_args)
        assert user_id in query_params

    async def test_user_admin_settings_is_frozen_dataclass(self):
        """Test that UserAdminSettings is immutable."""
        settings = UserAdminSettings(user_id=uuid.uuid4(), is_super_admin=False)

        with pytest.raises(AttributeError):
            settings.is_super_admin = True  # type: ignore


class TestAddUserAsAdmin:
    """Tests for the add_user_as_admin database function."""

    async def test_returns_existing_admin_when_user_is_already_admin(self):
        """Test that the existing settings are returned without changes."""
        user_id = uuid.uuid4()
        session = make_session(execute=[make_admin(user_id, is_super_admin=True)])

        result = await add_user_as_admin(user_id=user_id, session=session)

        assert result == UserAdminSettings(user_id=user_id, is_super_admin=True)

    async def test_does_not_write_when_user_is_already_admin(self):
        """Test that no row is added or flushed for an existing admin."""
        user_id = uuid.uuid4()
        session = make_session(execute=[make_admin(user_id)])

        await add_user_as_admin(user_id=user_id, session=session)

        session.add.assert_not_called()
        session.flush.assert_not_awaited()

    async def test_adds_admin_row_when_user_is_not_admin(self):
        """Test that a new AdminT row is added and flushed for a new admin."""
        user_id = uuid.uuid4()
        session = make_session(execute=[None])

        await add_user_as_admin(user_id=user_id, session=session)

        assert len(session.added) == 1
        new_admin = session.added[0]
        assert isinstance(new_admin, AdminT)
        assert new_admin.user_id == user_id
        session.flush.assert_awaited_once()

    async def test_new_admin_is_not_super_admin(self):
        """Test that a newly added admin gets is_super_admin=False."""
        user_id = uuid.uuid4()
        session = make_session(execute=[None])

        result = await add_user_as_admin(user_id=user_id, session=session)

        assert session.added[0].is_super_admin is False
        assert result == UserAdminSettings(user_id=user_id, is_super_admin=False)

    async def test_looks_up_existing_admin_by_user_id(self):
        """Test that the existence check filters on user_id."""
        user_id = uuid.uuid4()
        session = make_session(execute=[None])

        await add_user_as_admin(user_id=user_id, session=session)

        query_params = params(session.execute.await_args)
        assert user_id in query_params
