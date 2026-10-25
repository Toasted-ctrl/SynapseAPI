import pytest
from db_helpers import make_session, params, row

from database.models import get_models_by_expertise

pytestmark = pytest.mark.asyncio


class TestGetModelsByExpertise:
    async def test_returns_model_names(self):
        session = make_session(scalars=[[row(name="llama3"), row(name="gpt-5")]])

        result = await get_models_by_expertise(
            session=session, expertise="chat_completion"
        )

        assert result == ["llama3", "gpt-5"]
        assert params(session.scalars.await_args) == ["chat_completion"]

    async def test_returns_empty_list_when_none(self):
        session = make_session(scalars=[[]])

        assert (
            await get_models_by_expertise(session=session, expertise="translation")
            == []
        )
