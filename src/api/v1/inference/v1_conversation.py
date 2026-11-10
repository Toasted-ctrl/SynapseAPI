import json
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from httpx import ConnectError, ConnectTimeout
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.playground.models import AgentTools
from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.config import config
from core.logging import get_logger
from database.agent import AgentNotFoundError, get_agent_configuration
from database.message_threads import verify_or_get_thread_id
from database.providers import (
    ProviderConfiguration,
    UserProviderRegistry,
    get_all_provider_configurations,
)
from database.session import get_db_session
from providers.agent import (
    AgentInput,
    InvalidToolResultsError,
    NoPendingToolCallsError,
    PendingToolCallsError,
    build_agent_input,
    build_agent_model,
    run_agent,
    stream_agent,
)
from security.encryption import EncryptedSerializer
from tools.get_tools import get_agent_tools

from .helpers import ConversationSettings, ModelType, get_provider_configuration
from .models import ConversationCreate, ConversationRead, ToolCallRead

router = APIRouter()


log = get_logger()


encryption_serde = EncryptedSerializer(key=config.CHECKPOINT_ENCRYPTION_KEY)


def _serialize_event(event):
    """Recursively convert LangChain messages to dicts."""
    if isinstance(event, BaseMessage):
        return event.model_dump()
    elif isinstance(event, dict):
        return {k: _serialize_event(v) for k, v in event.items()}
    elif isinstance(event, list):
        return [_serialize_event(item) for item in event]
    return event


async def _resolve_settings(
    payload: ConversationCreate, session: AsyncSession, user_id: uuid.UUID
) -> ConversationSettings:
    """Returns the stored Agent's settings if agent_id is given, otherwise the inline ones."""
    if payload.agent_id:
        try:
            agent = await get_agent_configuration(
                session=session, user_id=user_id, agent_id=payload.agent_id
            )
        except AgentNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        return ConversationSettings(
            model=agent.model, system_prompt=agent.system_prompt, tools=agent.tools
        )
    if payload.model is None:
        raise ValueError("model is required when agent_id is omitted")
    return ConversationSettings(
        model=payload.model,
        system_prompt=payload.system_prompt,
        tools=payload.tools or AgentTools(),
        stop_sequence=payload.stop_sequence,
    )


def _check_client_tool_names(payload: ConversationCreate, tools: list[BaseTool]):
    """Raises a 422 if a client tool has the same name as a server tool."""
    clashes = {t.name for t in tools} & {t.name for t in payload.client_tools or []}
    if clashes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Client tool name(s) already used by server tools: {', '.join(sorted(clashes))}",
        )


@asynccontextmanager
async def _open_checkpointer():
    async with AsyncPostgresSaver.from_conn_string(
        conn_string=config.PG_CHECKPOINTER_URL, serde=encryption_serde
    ) as checkpointer:
        yield checkpointer


def _build_agent(
    payload: ConversationCreate,
    settings: ConversationSettings,
    prov: ProviderConfiguration,
    tools: list[BaseTool],
    checkpointer: AsyncPostgresSaver,
):
    return build_agent_model(
        langchain_con=prov.langchain_con,
        model=settings.model.name,
        base_url=prov.base_url,
        temperature=settings.model.parameters.temperature,
        top_k=settings.model.parameters.top_k,
        top_p=settings.model.parameters.top_p,
        max_output_tokens=settings.model.parameters.max_output_tokens,
        encrypted_api_key=prov.encrypted_api_key,
        tools=tools,
        system_prompt=settings.system_prompt,
        stop_sequence=settings.stop_sequence,
        checkpointer=checkpointer,
        client_tools=payload.client_tools,
    )


async def _resolve_agent_input(
    agent, payload: ConversationCreate, thread_id: uuid.UUID
) -> AgentInput:
    """Returns the input for the prompt or tool results, as an HTTPException if the
    thread's state doesn't allow it."""
    try:
        return await build_agent_input(
            agent=agent,
            thread_id=thread_id,
            prompt=payload.prompt,
            tool_results=payload.tool_results,
        )
    except PendingToolCallsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": f"{e} Send their results with tool_results first.",
                "pending_tool_calls": e.pending,
            },
        )
    except NoPendingToolCallsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except InvalidToolResultsError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )


@router.post(
    "/inference/conversation",
    tags=["Inference / Conversation"],
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
        settings: ConversationSettings = await _resolve_settings(
            payload=payload, session=session, user_id=user.id
        )

        upr: UserProviderRegistry = await get_all_provider_configurations(
            session=session, user_id=user.id
        )

        prov: ProviderConfiguration = await get_provider_configuration(
            user_provider_registry=upr,
            settings=settings,
            model_type=ModelType.CHAT_COMPLETION,
        )

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
            pr=upr,
        )

        _check_client_tool_names(payload=payload, tools=tools)

        async with _open_checkpointer() as checkpointer:
            agent = _build_agent(
                payload=payload,
                settings=settings,
                prov=prov,
                tools=tools,
                checkpointer=checkpointer,
            )
            agent_input = await _resolve_agent_input(
                agent=agent, payload=payload, thread_id=thread_id
            )
            result = await run_agent(
                agent=agent,
                agent_input=agent_input,
                thread_id=thread_id,
                user_id=user.id,
            )

        return ConversationRead(
            thread_id=thread_id,
            messages=[_serialize_event(m) for m in result["messages"]],
            pending_tool_calls=[
                ToolCallRead(**call) for call in result["pending_tool_calls"]
            ],
        )

    except ConnectTimeout, ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )


@router.post(
    "/inference/conversation/stream",
    tags=["Inference / Conversation"],
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
        settings: ConversationSettings = await _resolve_settings(
            payload=payload, session=session, user_id=user.id
        )

        upr: UserProviderRegistry = await get_all_provider_configurations(
            session=session, user_id=user.id
        )

        prov: ProviderConfiguration = await get_provider_configuration(
            user_provider_registry=upr,
            settings=settings,
            model_type=ModelType.CHAT_COMPLETION,
        )

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
            pr=upr,
        )

        _check_client_tool_names(payload=payload, tools=tools)

        # Resolved before streaming starts, so a thread in the wrong state gets a
        # real HTTP status. The checkpointer can't stay open until the stream
        # starts, so the stream opens its own.
        async with _open_checkpointer() as checkpointer:
            agent = _build_agent(
                payload=payload,
                settings=settings,
                prov=prov,
                tools=tools,
                checkpointer=checkpointer,
            )
            agent_input = await _resolve_agent_input(
                agent=agent, payload=payload, thread_id=thread_id
            )

        async def event_stream(thread_id: uuid.UUID):
            # Headers are already sent once streaming starts, so errors are
            # reported to the client as a final 'error' event instead of an HTTP status.
            try:
                async with _open_checkpointer() as checkpointer:
                    agent = _build_agent(
                        payload=payload,
                        settings=settings,
                        prov=prov,
                        tools=tools,
                        checkpointer=checkpointer,
                    )

                    async for event in stream_agent(
                        agent=agent,
                        agent_input=agent_input,
                        thread_id=thread_id,
                        user_id=user.id,
                    ):
                        yield f"data: {json.dumps(_serialize_event(event=event))}\n\n"

            except ConnectTimeout, ConnectError:
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

    except ConnectTimeout, ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )
