from fastapi import APIRouter, Depends, HTTPException, status
from httpx import ConnectError, ConnectTimeout, HTTPStatusError, ReadTimeout
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.playground.models import DecisionState, DecisionStateType
from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.agent import AgentNotFoundError, get_decision_agent_configuration
from database.providers import (
    ProviderConfiguration,
    UserProviderRegistry,
    get_all_provider_configurations,
)
from database.session import get_db_session
from providers.melious.system_one import get_decision

from .helpers import DecisionSettings, ModelType, get_provider_configuration
from .models import DecisionCreate, DecisionRead

router = APIRouter()


log = get_logger()


def _render_state(state: str | dict[str, str], config: DecisionState) -> str:
    """Validates the state against the Agent's state configuration and returns it as text.
    Fields are rendered one per line, in the order the Agent defines them."""
    if config.type == DecisionStateType.TEXT:
        if not isinstance(state, str) or not state.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This agent expects state as a non-empty string.",
            )
        return state

    if not isinstance(state, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This agent expects state as an object of named fields.",
        )
    names = [field.name for field in config.fields]
    missing = [name for name in names if name not in state]
    unknown = sorted(set(state).difference(names))
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown fields: {', '.join(unknown)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state, {'; '.join(problems)}.",
        )
    return "\n".join(f"{name}: {state[name]}" for name in names)


@router.post(
    "/inference/decision",
    tags=["Inference / Decision"],
    response_model=DecisionRead,
    description="Invokes a non-streaming decision from a stored decision agent.",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "State does not match the agent, or the provider is not configured or does not support decisions."
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Decision agent, provider or model does not exist."
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "Provider rejected the request."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Unable to connect to provider."
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "Provider did not respond in time."
        },
    },
)
async def decision(
    payload: DecisionCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> DecisionRead:
    try:
        agent = await get_decision_agent_configuration(
            session=session, user_id=user.id, agent_id=payload.agent_id
        )
    except AgentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    state = _render_state(state=payload.state, config=agent.state)

    upr: UserProviderRegistry = await get_all_provider_configurations(
        session=session, user_id=user.id
    )

    prov: ProviderConfiguration = await get_provider_configuration(
        user_provider_registry=upr,
        settings=DecisionSettings(model=agent.model),
        model_type=ModelType.SYSTEM_ONE,
    )

    # Melious is currently the only supported provider with System One models.
    if "melious" not in prov.base_url.lower() or prov.encrypted_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider does not support decisions: {prov.name}",
        )

    try:
        answers = await get_decision(
            base_url=prov.base_url,
            encrypted_api_key=prov.encrypted_api_key,
            model=agent.model.name,
            state=state,
            questions=agent.questions,
        )
    except ConnectTimeout, ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )
    except ReadTimeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Provider did not respond in time",
        )
    except HTTPStatusError as e:
        log.warning(
            f"Provider rejected decision request ({e.response.status_code}): {e.response.text}"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provider rejected the request with status {e.response.status_code}",
        )

    return DecisionRead(agent_id=agent.id, answers=answers)
