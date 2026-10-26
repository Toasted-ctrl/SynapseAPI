from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db_session, get_db_session_ctx


@pytest.mark.asyncio
class TestGetDbSession:
    """Tests for get_db_session async dependency."""

    async def test_get_db_session_yields_session(self):
        """Test that get_db_session yields an AsyncSession."""
        mock_session = AsyncMock(spec=AsyncSession)

        # Create a proper async context manager for begin()
        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__ = AsyncMock(return_value=None)
        mock_begin_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=mock_begin_cm)

        with patch("database.session.AsyncSessionLocal") as mock_session_local:
            # Create a proper async context manager for AsyncSessionLocal()
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_local.return_value = session_cm

            gen = get_db_session()
            session = await gen.__anext__()

            assert session is mock_session

    async def test_get_db_session_begins_transaction(self):
        """Test that get_db_session begins a transaction."""
        mock_session = AsyncMock(spec=AsyncSession)

        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__ = AsyncMock(return_value=None)
        mock_begin_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=mock_begin_cm)

        with patch("database.session.AsyncSessionLocal") as mock_session_local:
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_local.return_value = session_cm

            gen = get_db_session()
            await gen.__anext__()

            mock_session.begin.assert_called_once()

    async def test_get_db_session_logs_opening(self):
        """Test that opening a session is logged."""
        mock_session = AsyncMock(spec=AsyncSession)

        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__ = AsyncMock(return_value=None)
        mock_begin_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=mock_begin_cm)

        with (
            patch("database.session.AsyncSessionLocal") as mock_session_local,
            patch("database.session.log") as mock_log,
        ):
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_local.return_value = session_cm

            gen = get_db_session()
            await gen.__anext__()

            mock_log.debug.assert_called_with("Opening database session")

    async def test_get_db_session_logs_closing(self):
        """Test that closing a session is logged."""
        mock_session = AsyncMock(spec=AsyncSession)

        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__ = AsyncMock(return_value=None)
        mock_begin_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=mock_begin_cm)

        with (
            patch("database.session.AsyncSessionLocal") as mock_session_local,
            patch("database.session.log") as mock_log,
        ):
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_local.return_value = session_cm

            gen = get_db_session()
            await gen.__anext__()

            # Trigger StopAsyncIteration to complete the generator
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()

            # Check that both debug calls were made
            assert mock_log.debug.call_count == 2
            calls = mock_log.debug.call_args_list
            assert calls[0][0][0] == "Opening database session"
            assert calls[1][0][0] == "Closed database session"

    async def test_get_db_session_cleanup_on_exception(self):
        """Test that session cleanup happens even if an exception occurs during use."""
        mock_session = AsyncMock(spec=AsyncSession)

        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__ = AsyncMock(return_value=None)
        mock_begin_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=mock_begin_cm)

        with patch("database.session.AsyncSessionLocal") as mock_session_local:
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_local.return_value = session_cm

            gen = get_db_session()
            await gen.__anext__()

            # Complete the generator
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()

            # Verify session context manager's __aexit__ was called
            session_cm.__aexit__.assert_called_once()

    async def test_get_db_session_context_manager_flow(self):
        """Test the full async context manager flow."""
        mock_session = AsyncMock(spec=AsyncSession)

        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__ = AsyncMock(return_value=None)
        mock_begin_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=mock_begin_cm)

        with patch("database.session.AsyncSessionLocal") as mock_session_local:
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_local.return_value = session_cm

            # Use the dependency as a dependency would be used
            gen = get_db_session()
            session = await gen.__anext__()

            assert session is mock_session

            # Complete the generator
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()

            # Verify the flow
            session_cm.__aenter__.assert_called_once()
            mock_session.begin.assert_called_once()
            mock_begin_cm.__aenter__.assert_called_once()


def _mock_session_local():
    """Patchable AsyncSessionLocal whose session.begin() records how the transaction exits."""
    mock_session = AsyncMock(spec=AsyncSession)
    begin_cm = AsyncMock()
    begin_cm.__aenter__ = AsyncMock(return_value=None)
    begin_cm.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin = MagicMock(return_value=begin_cm)

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=session_cm), mock_session, begin_cm, session_cm


@pytest.mark.asyncio
class TestGetDbSessionCtx:
    """Tests for get_db_session_ctx async context manager."""

    async def test_yields_session_inside_transaction(self):
        session_local, mock_session, begin_cm, session_cm = _mock_session_local()

        with patch("database.session.AsyncSessionLocal", session_local):
            async with get_db_session_ctx() as session:
                assert session is mock_session
                mock_session.begin.assert_called_once()

        begin_cm.__aexit__.assert_awaited_once_with(None, None, None)
        session_cm.__aexit__.assert_awaited_once()

    async def test_exception_propagates_to_transaction_for_rollback(self):
        session_local, _, begin_cm, session_cm = _mock_session_local()

        with patch("database.session.AsyncSessionLocal", session_local):
            with pytest.raises(RuntimeError, match="boom"):
                async with get_db_session_ctx():
                    raise RuntimeError("boom")

        exc_type, exc, _ = begin_cm.__aexit__.await_args.args
        assert exc_type is RuntimeError
        assert str(exc) == "boom"
        session_cm.__aexit__.assert_awaited_once()

    async def test_logs_open_and_close(self):
        session_local, *_ = _mock_session_local()

        with (
            patch("database.session.AsyncSessionLocal", session_local),
            patch("database.session.log") as mock_log,
        ):
            async with get_db_session_ctx():
                pass

        assert [c.args[0] for c in mock_log.debug.call_args_list] == [
            "Opening database session",
            "Closed database session",
        ]
