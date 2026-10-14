import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from api.v1.playground.v1_models import (
    AgentConfigurationCreate,
    AgentConfigurationRead,
    AgentModelParameters,
    AgentTools,
)
from database.agent import (
    AgentConfigurationLimited,
    AgentNameTakenError,
    AgentNotFoundError,
    AgentReferenceError,
    create_agent_configuration,
    delete_agent_configuration,
    get_agent_configuration,
    list_agents_owned_by_user,
    update_agent_configuration,
)
from db_helpers import make_session, params, row



def make_config(**overrides) -> AgentConfigurationCreate:
    return AgentConfigurationCreate(**{
        "name": "helper",
        "system_prompt": "Be helpful.",
        "model": {"provider": "anthropic", "name": "claude-opus-5-5"},
        **overrides
    })


def stored_settings(config: AgentConfigurationCreate) -> dict:
    """The settings JSON create_agent_configuration/update_agent_configuration store for a configuration."""
    return {"version": 1, **config.model_dump(mode="json", exclude={"name"})}


def integrity_error(constraint: str) -> IntegrityError:
    return IntegrityError("INSERT ...", {}, Exception(f'duplicate key value violates unique constraint "{constraint}"'))


def mcp_row(mcp_id: uuid.UUID):
    return row(id=mcp_id, name="search", url="http://mcp", transport="streamable_http")


class TestAgentConfigurationModels:

    def test_missing_optional_fields_use_defaults(self):
        config = AgentConfigurationCreate.model_validate(
            {"name": "x", "model": {"provider": "p", "name": "m"}}
        )

        assert config.system_prompt is None
        assert config.model.parameters == AgentModelParameters()
        assert config.tools.mcp_ids == []

    @pytest.mark.parametrize("parameters", [
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_k": 0},
    ])
    def test_rejects_out_of_range_parameters(self, parameters):
        with pytest.raises(ValidationError):
            AgentModelParameters(**parameters)

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            make_config(name="")

    def test_create_ignores_id(self):
        config = AgentConfigurationCreate.model_validate(
            {"id": str(uuid.uuid4()), "name": "x", "model": {"provider": "p", "name": "m"}}
        )

        assert "id" not in config.model_dump()


@pytest.mark.asyncio
class TestCreateAgentConfiguration:

    async def test_stores_agent(self):
        user_id, mcp_id = uuid.uuid4(), uuid.uuid4()
        config = make_config(tools=AgentTools(mcp_ids=[mcp_id]))
        session = make_session(execute=[None, mcp_row(mcp_id)])

        result = await create_agent_configuration(session=session, user_id=user_id, config=config)

        stored = session.added[0]
        assert stored.owner_user_id == user_id
        assert stored.name == "helper"
        assert stored.settings == stored_settings(config)
        assert result == AgentConfigurationRead(id=stored.id, **config.model_dump())

    async def test_stored_settings_exclude_name_and_serialize_uuids(self):
        mcp_id = uuid.uuid4()
        config = make_config(tools=AgentTools(mcp_ids=[mcp_id]))
        session = make_session(execute=[None, mcp_row(mcp_id)])

        await create_agent_configuration(session=session, user_id=uuid.uuid4(), config=config)

        stored = session.added[0].settings
        assert "name" not in stored
        assert stored["version"] == 1
        assert stored["tools"]["mcp_ids"] == [str(mcp_id)]

    async def test_raises_when_name_taken(self):
        session = make_session(execute=[uuid.uuid4()])

        with pytest.raises(AgentNameTakenError, match="already exists"):
            await create_agent_configuration(session=session, user_id=uuid.uuid4(), config=make_config())
        assert session.added == []

    async def test_raises_when_mcp_missing(self):
        config = make_config(tools=AgentTools(mcp_ids=[uuid.uuid4()]))
        session = make_session(execute=[None, None])

        with pytest.raises(AgentReferenceError, match="does not exist"):
            await create_agent_configuration(session=session, user_id=uuid.uuid4(), config=config)
        assert session.added == []

    async def test_raises_name_taken_on_concurrent_insert(self):
        session = make_session(execute=[None])
        session.flush.side_effect = integrity_error("uq_agent_configs_owner_name")

        with pytest.raises(AgentNameTakenError, match="already exists"):
            await create_agent_configuration(session=session, user_id=uuid.uuid4(), config=make_config())

    async def test_reraises_other_integrity_errors(self):
        session = make_session(execute=[None])
        session.flush.side_effect = integrity_error("agent_configs_owner_user_id_fkey")

        with pytest.raises(IntegrityError):
            await create_agent_configuration(session=session, user_id=uuid.uuid4(), config=make_config())


@pytest.mark.asyncio
class TestGetAgentConfiguration:

    async def test_returns_configuration(self):
        user_id, agent_id = uuid.uuid4(), uuid.uuid4()
        config = make_config(model={
            "provider": "anthropic", "name": "claude-opus-5-5", "parameters": {"top_p": 0.9, "top_k": 40}
        })
        session = make_session(execute=[
            row(id=agent_id, owner_user_id=user_id, name="helper", settings=stored_settings(config))
        ])

        result = await get_agent_configuration(session=session, user_id=user_id, agent_id=agent_id)

        assert result == AgentConfigurationRead(id=agent_id, **config.model_dump())
        assert params(session.execute.await_args) == [agent_id, user_id]

    async def test_missing_optional_fields_use_defaults(self):
        agent = row(id=uuid.uuid4(), name="helper", settings={"model": {"provider": "p", "name": "m"}})
        session = make_session(execute=[agent])

        result = await get_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=agent.id)

        assert result.system_prompt is None
        assert result.model.parameters == AgentModelParameters()
        assert result.tools.mcp_ids == []

    async def test_tool_flags_missing_from_stored_tools_default_to_false(self):
        mcp_id = uuid.uuid4()
        settings = {"model": {"provider": "p", "name": "m"}, "tools": {"mcp_ids": [str(mcp_id)]}}
        session = make_session(execute=[row(id=uuid.uuid4(), name="helper", settings=settings)])

        result = await get_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4())

        assert result.tools == AgentTools(mcp_ids=[mcp_id])
        assert not result.tools.web_search

    async def test_unknown_fields_are_ignored(self):
        settings = {**stored_settings(make_config()), "removed_field": 1}
        session = make_session(execute=[row(id=uuid.uuid4(), name="helper", settings=settings)])

        result = await get_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4())

        assert "removed_field" not in result.model_dump()

    async def test_rejects_newer_version(self):
        settings = {**stored_settings(make_config()), "version": 99}
        session = make_session(execute=[row(id=uuid.uuid4(), name="helper", settings=settings)])

        with pytest.raises(ValueError, match="Unsupported agent settings version"):
            await get_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4())

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundError, match="Could not locate agent"):
            await get_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4())


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
class TestUpdateAgentConfiguration:

    async def test_replaces_configuration(self):
        agent_id = uuid.uuid4()
        agent = row(id=agent_id, name="helper", settings=stored_settings(make_config()))
        new_config = make_config(system_prompt="Be terse.")
        session = make_session(execute=[agent])

        result = await update_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=agent_id, config=new_config)

        assert agent.settings == stored_settings(new_config)
        assert result == AgentConfigurationRead(id=agent_id, **new_config.model_dump())
        session.flush.assert_awaited_once()

    async def test_unchanged_name_skips_availability_check(self):
        agent = row(id=uuid.uuid4(), name="helper", settings={})
        session = make_session(execute=[agent])

        await update_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=agent.id, config=make_config())

        assert session.execute.await_count == 1

    async def test_renames_agent(self):
        agent = row(id=uuid.uuid4(), name="helper", settings={})
        session = make_session(execute=[agent, None])

        result = await update_agent_configuration(
            session=session, user_id=uuid.uuid4(), agent_id=agent.id, config=make_config(name="renamed")
        )

        assert agent.name == "renamed"
        assert result.name == "renamed"

    async def test_raises_when_new_name_taken(self):
        agent = row(id=uuid.uuid4(), name="helper", settings={})
        session = make_session(execute=[agent, uuid.uuid4()])

        with pytest.raises(AgentNameTakenError, match="already exists"):
            await update_agent_configuration(
                session=session, user_id=uuid.uuid4(), agent_id=agent.id, config=make_config(name="taken")
            )
        assert agent.name == "helper"

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundError, match="Could not locate agent"):
            await update_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4(), config=make_config())


@pytest.mark.asyncio
class TestDeleteAgentConfiguration:

    async def test_deletes_agent(self):
        agent = row(id=uuid.uuid4())
        session = make_session(execute=[agent])

        await delete_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=agent.id)

        session.delete.assert_awaited_once_with(agent)

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(AgentNotFoundError, match="Could not locate agent"):
            await delete_agent_configuration(session=session, user_id=uuid.uuid4(), agent_id=uuid.uuid4())
