import uuid

import pytest
from db_helpers import make_session, params

from database.agent_ownership import (
    AgentNotFoundOrNotOwnedError,
    ensure_agent_owned_by_user,
)

pytestmark = pytest.mark.asyncio


class TestEnsureAgentOwnedByUser:
    async def test_passes_when_agent_owned_by_user(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        session = make_session(execute=[agent_id])

        # Should not raise
        await ensure_agent_owned_by_user(
            user_id=owner_user_id, agent_id=agent_id, session=session
        )

    async def test_raises_when_agent_not_found(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundOrNotOwnedError) as exc_info:
            await ensure_agent_owned_by_user(
                user_id=owner_user_id, agent_id=agent_id, session=session
            )

        assert str(agent_id) in str(exc_info.value)
        assert str(owner_user_id) in str(exc_info.value)

    async def test_raises_when_agent_not_owned_by_user(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        # Agent exists but owned by a different user
        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundOrNotOwnedError) as exc_info:
            await ensure_agent_owned_by_user(
                user_id=owner_user_id, agent_id=agent_id, session=session
            )

        assert str(agent_id) in str(exc_info.value)
        assert str(owner_user_id) in str(exc_info.value)

    async def test_filters_on_agent_id_and_owner_user_id(self):
        owner_user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        session = make_session(execute=[agent_id])

        await ensure_agent_owned_by_user(
            user_id=owner_user_id, agent_id=agent_id, session=session
        )

        bound = params(session.execute.await_args)
        assert agent_id in bound
        assert owner_user_id in bound
