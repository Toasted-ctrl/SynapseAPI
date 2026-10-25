import json
import uuid
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from httpx import ConnectError, ConnectTimeout
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.playground.models import AgentModel, AgentTools
from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.config import config
from core.logging import get_logger
from core.model_types import model_config
from database.agent import AgentNotFoundError, get_agent_configuration
from database.message_threads import verify_or_get_thread_id
from database.providers import (
    ProviderConfiguration,
    UserProviderRegistry,
    get_all_provider_configurations,
)
from database.session import get_db_session
from providers.agent import build_agent_model, run_agent, stream_agent
from tools.get_tools import get_agent_tools

from .models import ConversationCreate, ConversationRead

router = APIRouter()


log = get_logger()


def _serialize_event(event):
    """Recursively convert LangChain messages to dicts."""
    if isinstance(event, BaseMessage):
        return event.model_dump()
    elif isinstance(event, dict):
        return {k: _serialize_event(v) for k, v in event.items()}
    elif isinstance(event, list):
        return [_serialize_event(item) for item in event]
    return event


@dataclass(frozen=True)
class _ConversationSettings:
    model: AgentModel
    system_prompt: str | None
    tools: AgentTools


async def _resolve_settings(
    payload: ConversationCreate, session: AsyncSession, user_id: uuid.UUID
) -> _ConversationSettings:
    """Returns the stored Agent's settings if agent_id is given, otherwise the inline ones.
    Only stored Agents have a system prompt."""
    if payload.agent_id:
        try:
            agent = await get_agent_configuration(
                session=session, user_id=user_id, agent_id=payload.agent_id
            )
        except AgentNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        return _ConversationSettings(
            model=agent.model, system_prompt=agent.system_prompt, tools=agent.tools
        )
    return _ConversationSettings(
        model=payload.model, system_prompt=None, tools=payload.tools or AgentTools()
    )


@router.post(
    "/conversation",
    tags=["Conversation"],
    response_model=ConversationRead,
    description=(
        "Invokes a non-streaming response from an agent (by agent_id), or a "
        "response from a model if an agent_id is not provided."
    ),
)
async def conversation(
    payload: ConversationCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationRead:
    try:
        settings: _ConversationSettings = await _resolve_settings(
            payload=payload, session=session, user_id=user.id
        )

        if settings.model.name not in await model_config.chat_completion_models():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model not supported by Provider",
            )

        pr: UserProviderRegistry = await get_all_provider_configurations(
            session=session, user_id=user.id
        )

        if settings.model.provider in pr.not_configured:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {settings.model.provider} not configured",
            )

        prov: ProviderConfiguration = getattr(pr, settings.model.provider)

        thread_id: uuid.UUID = await verify_or_get_thread_id(
            session=session, user_id=user.id, thread_id=payload.thread_id
        )

        tools = await get_agent_tools(
            mcp_ids=settings.tools.mcp_ids,
            session=session,
            web_search=settings.tools.web_search,
            user_vs_files=settings.tools.user_vs_files,
            user_vs_memories=settings.tools.user_vs_memories,
            user_vs_skills=settings.tools.user_vs_skills,
            user_id=user.id,
            pr=pr,
        )

        async with AsyncPostgresSaver.from_conn_string(
            conn_string=config.PG_CHECKPOINTER_URL
        ) as checkpointer:
            agent = build_agent_model(
                langchain_con=prov.langchain_con,
                model=settings.model.name,
                base_url=prov.base_url,
                temperature=settings.model.parameters.temperature,
                top_k=settings.model.parameters.top_k,
                top_p=settings.model.parameters.top_p,
                encrypted_api_key=prov.encrypted_api_key,
                tools=tools,
                system_prompt=settings.system_prompt,
                checkpointer=checkpointer,
            )

            result = await run_agent(
                agent=agent, prompt=payload.prompt, thread_id=thread_id, user_id=user.id
            )

        return ConversationRead(
            thread_id=thread_id,
            messages=[_serialize_event(m) for m in result["messages"]],
        )

    except (ConnectTimeout, ConnectError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )


@router.post(
    "/conversation/stream",
    tags=["Conversation"],
    response_class=StreamingResponse,
    description=(
        "Invokes a streaming response from an agent (by agent_id), or a "
        "streaming response from a model if an agent_id is not provided."
    ),
)
async def conversation_stream(
    payload: ConversationCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    try:
        settings: _ConversationSettings = await _resolve_settings(
            payload=payload, session=session, user_id=user.id
        )

        if settings.model.name not in await model_config.chat_completion_models():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model not supported by Provider",
            )

        pr: UserProviderRegistry = await get_all_provider_configurations(
            session=session, user_id=user.id
        )

        if settings.model.provider in pr.not_configured:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {settings.model.provider} not configured",
            )

        prov: ProviderConfiguration = getattr(pr, settings.model.provider)

        thread_id: uuid.UUID = await verify_or_get_thread_id(
            session=session, user_id=user.id, thread_id=payload.thread_id
        )

        tools = await get_agent_tools(
            mcp_ids=settings.tools.mcp_ids,
            session=session,
            web_search=settings.tools.web_search,
            user_vs_files=settings.tools.user_vs_files,
            user_vs_memories=settings.tools.user_vs_memories,
            user_vs_skills=settings.tools.user_vs_skills,
            user_id=user.id,
            pr=pr,
        )

        async def event_stream(thread_id: uuid.UUID):
            # Headers are already sent once streaming starts, so errors are
            # reported to the client as a final 'error' event instead of an HTTP status.
            try:
                async with AsyncPostgresSaver.from_conn_string(
                    conn_string=config.PG_CHECKPOINTER_URL
                ) as checkpointer:
                    agent = build_agent_model(
                        langchain_con=prov.langchain_con,
                        model=settings.model.name,
                        base_url=prov.base_url,
                        temperature=settings.model.parameters.temperature,
                        top_k=settings.model.parameters.top_k,
                        top_p=settings.model.parameters.top_p,
                        encrypted_api_key=prov.encrypted_api_key,
                        tools=tools,
                        system_prompt=settings.system_prompt,
                        checkpointer=checkpointer,
                    )

                    async for event in stream_agent(
                        agent=agent,
                        prompt=payload.prompt,
                        thread_id=thread_id,
                        user_id=user.id,
                    ):
                        yield f"data: {json.dumps(_serialize_event(event=event))}\n\n"

            except (ConnectTimeout, ConnectError):
                error = {
                    "type": "error",
                    "data": {"detail": "Unable to connect to Provider"},
                }
                yield f"data: {json.dumps(error)}\n\n"
            except Exception as e:
                log.exception(e)
                error = {
                    "type": "error",
                    "data": {"detail": "Agent run failed unexpectedly"},
                }
                yield f"data: {json.dumps(error)}\n\n"

        return StreamingResponse(
            event_stream(thread_id=thread_id),
            media_type="text/event-stream",
            headers={"X-Thread-ID": str(thread_id)},
        )

    except (ConnectTimeout, ConnectError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )
