import uuid

import pytest
from db_helpers import make_session, params, row

from database.agent_sharing import (
    AgentAlreadySharedError,
    AgentNotFoundOrNotOwnedError,
    AgentOwnedByUserError,
    UserNotFoundError,
    delete_shared_agent_from_user,
    get_shared_users_by_agent_id,
    share_agent_with_user_id,
)
from database.schemas.agent_sharing import AgentSharingT

pytestmark = pytest.mark.asyncio


def _agent_row(owner_user_id: uuid.UUID, agent_id: uuid.UUID = None):
    """Helper to create an AgentConfigT row."""
    return row(
        id=agent_id or uuid.uuid4(),
        owner_user_id=owner_user_id,
        name="Test Agent",
        description="Test Description",
        settings={},
        agent_type="Conversational",
    )


def _user_row(user_id: uuid.UUID = None):
    """Helper to create a UsersT row."""
    return row(
        id=user_id or uuid.uuid4(),
        person_id=uuid.uuid4(),
        api_key_id=uuid.uuid4(),
    )


class TestShareAgentWithUserId:
    async def test_shares_agent_with_user(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        share_with_user_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        user = _user_row(share_with_user_id)

        # First execute for agent lookup, second for user lookup
        session = make_session(execute=[agent, user])

        result = await share_agent_with_user_id(
            owner_user_id=owner_user_id,
            agent_id=agent_id,
            share_with_user_id=share_with_user_id,
            session=session,
        )

        assert isinstance(result, AgentSharingT)
        assert result.agent_id == agent_id
        assert result.user_id == share_with_user_id
        assert result.owner_user_id == owner_user_id
        session.add.assert_called_once()
        session.flush.assert_awaited_once()
        session.refresh.assert_awaited_once()

    async def test_raises_when_agent_not_found(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        share_with_user_id = uuid.uuid4()

        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundOrNotOwnedError) as exc_info:
            await share_agent_with_user_id(
                owner_user_id=owner_user_id,
                agent_id=agent_id,
                share_with_user_id=share_with_user_id,
                session=session,
            )

        assert str(agent_id) in str(exc_info.value)
        assert str(owner_user_id) in str(exc_info.value)
        session.add.assert_not_called()

    async def test_raises_when_agent_not_owned_by_user(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        share_with_user_id = uuid.uuid4()

        # Agent doesn't exist for this owner
        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundOrNotOwnedError) as exc_info:
            await share_agent_with_user_id(
                owner_user_id=owner_user_id,
                agent_id=agent_id,
                share_with_user_id=share_with_user_id,
                session=session,
            )

        assert str(agent_id) in str(exc_info.value)
        assert str(owner_user_id) in str(exc_info.value)
        session.add.assert_not_called()

    async def test_raises_when_sharing_with_owner(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        user = _user_row(owner_user_id)

        session = make_session(execute=[agent, user])

        with pytest.raises(AgentOwnedByUserError) as exc_info:
            await share_agent_with_user_id(
                owner_user_id=owner_user_id,
                agent_id=agent_id,
                share_with_user_id=owner_user_id,
                session=session,
            )

        assert "cannot be shared with their owner" in str(exc_info.value)
        session.add.assert_not_called()

    async def test_raises_when_user_not_found(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        share_with_user_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        session = make_session(execute=[agent, None])

        with pytest.raises(UserNotFoundError) as exc_info:
            await share_agent_with_user_id(
                owner_user_id=owner_user_id,
                agent_id=agent_id,
                share_with_user_id=share_with_user_id,
                session=session,
            )

        assert str(share_with_user_id) in str(exc_info.value)
        session.add.assert_not_called()

    async def test_raises_when_already_shared(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        share_with_user_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        user = _user_row(share_with_user_id)

        # Mock IntegrityError to simulate duplicate key
        from sqlalchemy.exc import IntegrityError

        session = make_session(execute=[agent, user])

        # Override flush to raise IntegrityError
        async def _flush_with_error():
            raise IntegrityError("mock", "params", " Duplicate entry")

        session.flush = _flush_with_error

        with pytest.raises(AgentAlreadySharedError) as exc_info:
            await share_agent_with_user_id(
                owner_user_id=owner_user_id,
                agent_id=agent_id,
                share_with_user_id=share_with_user_id,
                session=session,
            )

        assert str(agent_id) in str(exc_info.value)
        assert str(share_with_user_id) in str(exc_info.value)

    async def test_filters_on_agent_id_and_owner_user_id(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        share_with_user_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        user = _user_row(share_with_user_id)

        session = make_session(execute=[agent, user])

        await share_agent_with_user_id(
            owner_user_id=owner_user_id,
            agent_id=agent_id,
            share_with_user_id=share_with_user_id,
            session=session,
        )

        # Check first execute call (agent lookup)
        first_call = session.execute.await_args_list[0]
        bound = params(first_call)
        assert agent_id in bound
        assert owner_user_id in bound


class TestGetSharedUsersByAgentId:
    async def test_returns_list_of_user_ids(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        user_id_1 = uuid.uuid4()
        user_id_2 = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        # scalars returns the user_id values directly (not row objects)
        sharing_user_ids = [user_id_1, user_id_2]

        # First execute for ownership check, then scalars for sharing lookup
        session = make_session(
            execute=[agent],
            scalars=[sharing_user_ids],
        )

        result = await get_shared_users_by_agent_id(
            owner_user_id=owner_user_id,
            agent_id=agent_id,
            session=session,
        )

        assert result == [user_id_1, user_id_2]

    async def test_returns_empty_list_when_no_shares(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)

        session = make_session(
            execute=[agent],
            scalars=[[]],
        )

        result = await get_shared_users_by_agent_id(
            owner_user_id=owner_user_id,
            agent_id=agent_id,
            session=session,
        )

        assert result == []

    async def test_raises_when_agent_not_owned_by_user(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundOrNotOwnedError):
            await get_shared_users_by_agent_id(
                owner_user_id=owner_user_id,
                agent_id=agent_id,
                session=session,
            )

    async def test_filters_on_agent_id(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)

        session = make_session(
            execute=[agent],
            scalars=[[]],
        )

        await get_shared_users_by_agent_id(
            owner_user_id=owner_user_id,
            agent_id=agent_id,
            session=session,
        )

        # Check the scalars call (sharing lookup)
        bound = params(session.scalars.await_args)
        assert agent_id in bound


class TestDeleteSharedAgentFromUser:
    async def test_deletes_sharing_record(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        shared_with_user_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        sharing = row(
            agent_id=agent_id,
            owner_user_id=owner_user_id,
            user_id=shared_with_user_id,
        )

        # First execute for ownership check, second for sharing lookup
        session = make_session(execute=[agent, sharing])

        result = await delete_shared_agent_from_user(
            owner_user_id=owner_user_id,
            shared_with_user_id=shared_with_user_id,
            agent_id=agent_id,
            session=session,
        )

        assert result is True
        session.delete.assert_awaited_once_with(sharing)
        session.flush.assert_awaited_once()

    async def test_raises_when_sharing_not_found(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        shared_with_user_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)

        session = make_session(execute=[agent, None])

        with pytest.raises(AgentNotFoundOrNotOwnedError) as exc_info:
            await delete_shared_agent_from_user(
                owner_user_id=owner_user_id,
                shared_with_user_id=shared_with_user_id,
                agent_id=agent_id,
                session=session,
            )

        assert str(agent_id) in str(exc_info.value)
        assert str(shared_with_user_id) in str(exc_info.value)
        session.delete.assert_not_called()

    async def test_raises_when_agent_not_owned_by_user(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        shared_with_user_id = uuid.uuid4()

        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundOrNotOwnedError):
            await delete_shared_agent_from_user(
                owner_user_id=owner_user_id,
                shared_with_user_id=shared_with_user_id,
                agent_id=agent_id,
                session=session,
            )

        session.delete.assert_not_called()

    async def test_filters_on_all_three_ids(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        shared_with_user_id = uuid.uuid4()

        agent = _agent_row(owner_user_id, agent_id)
        sharing = row(
            agent_id=agent_id,
            owner_user_id=owner_user_id,
            user_id=shared_with_user_id,
        )

        session = make_session(execute=[agent, sharing])

        await delete_shared_agent_from_user(
            owner_user_id=owner_user_id,
            shared_with_user_id=shared_with_user_id,
            agent_id=agent_id,
            session=session,
        )

        # Check the second execute call (sharing lookup)
        second_call = session.execute.await_args_list[1]
        bound = params(second_call)
        assert agent_id in bound
        assert owner_user_id in bound
        assert shared_with_user_id in bound
