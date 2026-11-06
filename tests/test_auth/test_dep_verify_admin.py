import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import VerifiedAdmin, depends_verify_admin
from auth.dep_verify_user import VerifiedUser
from database.admin_users import NoAdminError

pytestmark = pytest.mark.asyncio


class TestDependsVerifyAdmin:
    """Tests for the depends_verify_admin dependency function."""

    @pytest.fixture
    def verified_user(self):
        """Create a sample VerifiedUser."""
        return VerifiedUser(id=uuid.uuid4())

    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession."""
        return MagicMock(spec=AsyncSession)

    @pytest.fixture
    def mock_get_admin_user_by_user_id(self):
        """Mock the get_admin_user_by_user_id function."""
        with patch(
            "auth.dep_verify_admin.get_admin_user_by_user_id",
            new=AsyncMock(),
        ) as mock:
            yield mock

    async def test_returns_verified_admin_on_success(
        self, verified_user, mock_session, mock_get_admin_user_by_user_id
    ):
        """Test that a VerifiedAdmin is returned when user is an admin."""
        user_id = verified_user.id
        mock_get_admin_user_by_user_id.return_value = user_id

        result = await depends_verify_admin(user=verified_user, session=mock_session)

        assert isinstance(result, VerifiedAdmin)
        assert result.id == user_id
        mock_get_admin_user_by_user_id.assert_awaited_once_with(
            user_id=user_id, session=mock_session
        )

    async def test_raises_http_exception_when_user_is_not_admin(
        self, verified_user, mock_session, mock_get_admin_user_by_user_id
    ):
        """Test that HTTPException is raised when user is not an admin."""
        error_message = f"User {verified_user.id} is not an Admin."
        mock_get_admin_user_by_user_id.side_effect = NoAdminError(error_message)

        with pytest.raises(HTTPException) as exc_info:
            await depends_verify_admin(user=verified_user, session=mock_session)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert str(exc_info.value.detail) == error_message
        mock_get_admin_user_by_user_id.assert_awaited_once_with(
            user_id=verified_user.id, session=mock_session
        )

    async def test_passes_correct_user_id_to_database_function(
        self, verified_user, mock_session, mock_get_admin_user_by_user_id
    ):
        """Test that the correct user_id is passed to get_admin_user_by_user_id."""
        expected_user_id = verified_user.id
        mock_get_admin_user_by_user_id.return_value = expected_user_id

        await depends_verify_admin(user=verified_user, session=mock_session)

        mock_get_admin_user_by_user_id.assert_awaited_once()
        call_args = mock_get_admin_user_by_user_id.await_args
        assert call_args.kwargs["user_id"] == expected_user_id
        assert call_args.kwargs["session"] == mock_session

    async def test_returns_verified_admin_with_correct_id(
        self, verified_user, mock_session, mock_get_admin_user_by_user_id
    ):
        """Test that VerifiedAdmin contains the correct user_id."""
        returned_user_id = uuid.uuid4()
        mock_get_admin_user_by_user_id.return_value = returned_user_id

        result = await depends_verify_admin(user=verified_user, session=mock_session)

        assert result.id == returned_user_id

    async def test_verified_admin_is_frozen_dataclass(self):
        """Test that VerifiedAdmin is a frozen dataclass."""
        admin = VerifiedAdmin(id=uuid.uuid4())

        with pytest.raises(AttributeError):
            admin.id = uuid.uuid4()  # type: ignore
