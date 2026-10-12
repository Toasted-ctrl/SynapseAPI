import uuid

import pytest
from pydantic import ValidationError

from database.agent import (
    AgentConfiguration,
    AgentConfigurationLimited,
    AgentSettings,
    AgentTools,
    create_agent,
    delete_agent,
    get_agent,
    list_agents_owned_by_user,
    update_agent,
)
from db_helpers import make_session, params, row



def make_settings(**overrides) -> AgentSettings:
    return AgentSettings(**{
        "system_prompt": "Be helpful.",
        "provider_name": "anthropic",
        "provider_model": "claude-opus-5-5",
        **overrides
    })


def mcp_row(mcp_id: uuid.UUID):
    return row(id=mcp_id, name="search", url="http://mcp", transport="streamable_http")


class TestAgentSettings:

    def test_round_trips_through_json(self):
        settings = make_settings(tools=AgentTools(mcp_ids=[uuid.uuid4()]))

        assert AgentSettings.model_validate(settings.model_dump(mode="json")) == settings

    def test_dump_serializes_uuids_as_strings(self):
        mcp_id = uuid.uuid4()

        dumped = make_settings(tools=AgentTools(mcp_ids=[mcp_id])).model_dump(mode="json")

        assert dumped["tools"]["mcp_ids"] == [str(mcp_id)]
        assert dumped["version"] == 1

    def test_missing_optional_fields_use_defaults(self):
        settings = AgentSettings.model_validate(
            {"system_prompt": "x", "provider_name": "p", "provider_model": "m"}
        )

        assert settings.version == 1
        assert settings.model_params == {}
        assert settings.tools.mcp_ids == []

    def test_model_params_round_trip(self):
        settings = make_settings(model_params={"top_p": 0.9, "top_k": 40})

        assert AgentSettings.model_validate(settings.model_dump(mode="json")).model_params == {"top_p": 0.9, "top_k": 40}

    def test_unknown_fields_are_ignored(self):
        settings = AgentSettings.model_validate(
            {"system_prompt": "x", "provider_name": "p", "provider_model": "m", "removed_field": 1}
        )

        assert "removed_field" not in settings.model_dump()

    def test_rejects_newer_version(self):
        with pytest.raises(ValidationError, match="Unsupported agent settings version"):
            AgentSettings.model_validate(
                {"version": 99, "system_prompt": "x", "provider_name": "p", "provider_model": "m"}
            )


@pytest.mark.asyncio
class TestCreateAgent:

    async def test_stores_agent(self):
        user_id, mcp_id = uuid.uuid4(), uuid.uuid4()
        settings = make_settings(tools=AgentTools(mcp_ids=[mcp_id]))
        session = make_session(execute=[None, mcp_row(mcp_id)])

        result = await create_agent(session=session, user_id=user_id, name="helper", settings=settings)

        stored = session.added[0]
        assert stored.owner_user_id == user_id
        assert stored.name == "helper"
        assert stored.settings == settings.model_dump(mode="json")
        assert result == AgentConfiguration(id=stored.id, name="helper", settings=settings)

    async def test_raises_when_name_taken(self):
        session = make_session(execute=[uuid.uuid4()])

        with pytest.raises(ValueError, match="already exists"):
            await create_agent(session=session, user_id=uuid.uuid4(), name="helper", settings=make_settings())
        assert session.added == []

    async def test_raises_when_mcp_missing(self):
        settings = make_settings(tools=AgentTools(mcp_ids=[uuid.uuid4()]))
        session = make_session(execute=[None, None])

        with pytest.raises(ValueError, match="does not exist"):
            await create_agent(session=session, user_id=uuid.uuid4(), name="helper", settings=settings)
        assert session.added == []


@pytest.mark.asyncio
class TestGetAgent:

    async def test_returns_configuration(self):
        user_id, agent_id = uuid.uuid4(), uuid.uuid4()
        settings = make_settings()
        session = make_session(execute=[
            row(id=agent_id, owner_user_id=user_id, name="helper", settings=settings.model_dump(mode="json"))
        ])

        result = await get_agent(session=session, user_id=user_id, agent_id=agent_id)

        assert result == AgentConfiguration(id=agent_id, name="helper", settings=settings)
        assert params(session.execute.await_args) == [agent_id, user_id]

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(ValueError, match="Could not locate agent"):
            await get_agent(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4())


@pytest.mark.asyncio
class TestListAgentsOwnedByUser:

    async def test_returns_id_and_name_of_all_agents(self):
        user_id, a, b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        session = make_session(execute=[[row(id=a, name="alpha"), row(id=b, name="beta")]])

        result = await list_agents_owned_by_user(session=session, user_id=user_id)

        assert result == [
            AgentConfigurationLimited(id=a, name="alpha"),
            AgentConfigurationLimited(id=b, name="beta")
        ]
        assert params(session.execute.await_args) == [user_id]

    async def test_selects_only_id_and_name(self):
        session = make_session(execute=[[]])

        await list_agents_owned_by_user(session=session, user_id=uuid.uuid4())

        statement = session.execute.await_args.args[0]
        assert [column.name for column in statement.selected_columns] == ["id", "name"]

    async def test_returns_empty_list_when_none(self):
        session = make_session(execute=[[]])

        assert await list_agents_owned_by_user(session=session, user_id=uuid.uuid4()) == []


@pytest.mark.asyncio
class TestUpdateAgent:

    async def test_replaces_settings(self):
        agent_id = uuid.uuid4()
        agent = row(id=agent_id, name="helper", settings=make_settings().model_dump(mode="json"))
        new_settings = make_settings(system_prompt="Be terse.")
        session = make_session(execute=[agent])

        result = await update_agent(session=session, user_id=uuid.uuid4(), agent_id=agent_id, settings=new_settings)

        assert agent.settings == new_settings.model_dump(mode="json")
        assert result.settings == new_settings
        session.flush.assert_awaited_once()

    async def test_renames_agent(self):
        agent = row(id=uuid.uuid4(), name="helper", settings={})
        session = make_session(execute=[agent, None])

        result = await update_agent(
            session=session, user_id=uuid.uuid4(), agent_id=agent.id, settings=make_settings(), name="renamed"
        )

        assert agent.name == "renamed"
        assert result.name == "renamed"

    async def test_raises_when_new_name_taken(self):
        agent = row(id=uuid.uuid4(), name="helper", settings={})
        session = make_session(execute=[agent, uuid.uuid4()])

        with pytest.raises(ValueError, match="already exists"):
            await update_agent(
                session=session, user_id=uuid.uuid4(), agent_id=agent.id, settings=make_settings(), name="taken"
            )
        assert agent.name == "helper"

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(ValueError, match="Could not locate agent"):
            await update_agent(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4(), settings=make_settings())


@pytest.mark.asyncio
class TestDeleteAgent:

    async def test_deletes_agent(self):
        agent = row(id=uuid.uuid4())
        session = make_session(execute=[agent])

        await delete_agent(session=session, user_id=uuid.uuid4(), agent_id=agent.id)

        session.delete.assert_awaited_once_with(agent)

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(ValueError, match="Could not locate agent"):
            await delete_agent(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4())
