import pytest
from db_helpers import make_session, params, row

from database.models import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelWithExpertise,
    add_model_with_expertise,
    delete_model_by_name,
    get_models_by_expertise,
)

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


class TestAddModelWithExpertise:
    async def test_adds_new_model(self):
        session = make_session(execute=[None])

        result = await add_model_with_expertise(
            session=session, model_name="llama3", expertise="chat_completion"
        )

        assert isinstance(result, ModelWithExpertise)
        assert result.model_name == "llama3"
        assert result.expertise == "chat_completion"
        assert result.created_date is not None
        assert len(session.added) == 1
        assert session.added[0].name == "llama3"
        assert session.added[0].expertise == "chat_completion"

    async def test_raises_error_for_duplicate(self):
        existing = row(name="llama3", expertise="chat_completion")
        session = make_session(execute=[existing])

        with pytest.raises(ModelAlreadyExistsError) as exc_info:
            await add_model_with_expertise(
                session=session, model_name="llama3", expertise="chat_completion"
            )

        assert "llama3" in str(exc_info.value)
        assert "already exists" in str(exc_info.value)
        assert session.added == []

    async def test_same_name_different_expertise_raises(self):
        # The model name is the primary key, so one name can only have one expertise.
        existing = row(name="llama3", expertise="chat_completion")
        session = make_session(execute=[existing])

        with pytest.raises(ModelAlreadyExistsError) as exc_info:
            await add_model_with_expertise(
                session=session, model_name="llama3", expertise="translation"
            )

        assert "chat_completion" in str(exc_info.value)
        assert session.added == []

    async def test_duplicate_check_filters_on_name_only(self):
        session = make_session(execute=[None])

        await add_model_with_expertise(
            session=session, model_name="llama3", expertise="translation"
        )

        assert params(session.execute.call_args) == ["llama3"]

    async def test_refreshes_to_load_server_defaults(self):
        session = make_session(execute=[None])

        await add_model_with_expertise(
            session=session, model_name="llama3", expertise="chat_completion"
        )

        session.refresh.assert_awaited_once_with(session.added[0])


class TestDeleteModelByName:
    async def test_deletes_model(self):
        existing = row(name="llama3", expertise="chat_completion", created_date=None)
        session = make_session(execute=[existing])

        result = await delete_model_by_name(session=session, model_name="llama3")

        session.delete.assert_awaited_once_with(existing)
        assert result.model_name == "llama3"
        assert result.expertise == "chat_completion"

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(ModelNotFoundError, match="llama3 does not exist"):
            await delete_model_by_name(session=session, model_name="llama3")

        session.delete.assert_not_awaited()

    async def test_filters_on_name(self):
        existing = row(name="llama3", expertise="chat_completion", created_date=None)
        session = make_session(execute=[existing])

        await delete_model_by_name(session=session, model_name="llama3")

        assert params(session.execute.await_args) == ["llama3"]
