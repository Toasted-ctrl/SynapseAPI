import base64
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from urllib.parse import urlencode

from auth.dep_verify_client import VerifiedClient, verify_client_from_application_id
from core.config import config
from core.logging import get_logger
from database.client import ApplicationClient, get_client_from_client_id
from database.session import get_db_session
from database.user_sessions import post_session
from security.encryption import decrypt
from security.google import verify_google_token, VerifiedGoogleUser, exchange_google_code, ExchangedGoogleCode
from security.hmac import hash_hmac, is_valid_hmac
from setup.application_user import VerifiedApplicationUser, get_or_create_application_user


log = get_logger()


router = APIRouter()


ENABLED = config.ENABLE_GOOGLE_LOGIN


@router.get(
    "/auth/google/login",
    description=(
        "WARNING!!! This method will not work when called through the documentation. "
        "Please call this path through a browser directly.\n"
        "\nRedirect the user to Google's OAuth2 consent screen."
    ),
    response_class=RedirectResponse,
    tags=["Auth"]
)
async def google_login(
    application_id: str,
    session: AsyncSession = Depends(get_db_session)
) -> RedirectResponse:

    # TODO: For frontend apps the api key is actually the application ID.
    # We'll want to change that and make it more clear in the schema.

    client: VerifiedClient = await verify_client_from_application_id(
        application_id=application_id,
        session=session
    )

    state_data = json.dumps({"client_id": str(client.id)})
    state = base64.urlsafe_b64encode(state_data.encode()).decode()
    state_signature = hash_hmac(content=state, key=config.GOOGLE_HMAC_SECRET)
    signed_state = f"{state}.{state_signature}"

    log.debug(f"Created signed state for Client '{client.id}'...")

    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": config.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": signed_state
    }

    url = f"{config.GOOGLE_AUTH_URL}?{urlencode(params)}"

    log.debug(f"Redirecting User for Client '{client.id}' to Google's consent screen...")

    return RedirectResponse(url=url)


@router.get(
    "/auth/google/callback",
    description=(
        "WARNING!!! This is a callback method for Google's Login, "
        "and will not work when called directly through the browser.\n"
        "\nNOTE: This method will store a cookie on the user's browser."
    ),
    response_class=RedirectResponse,
    tags=["Auth"]
)
async def google_callback(
    code: str,
    state: str,
    session: AsyncSession = Depends(get_db_session)
) -> RedirectResponse:

    log.debug("Receiving Callback from Google's OAuth service...")
    try:
        state_data, state_signature = state.rsplit(".", 1)
        log.debug("Received valid state format from Google...")
    except ValueError:
        log.warning("Received invalid state format from Google...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid State format"
        )

    if not is_valid_hmac(
        provided_hmac=state_signature,
        expected_hmac=hash_hmac(
            content=state_data,
            key=config.GOOGLE_HMAC_SECRET
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid State"
        )

    decoded_state = json.loads(base64.urlsafe_b64decode(state_data))
    raw_client_id = decoded_state.get("client_id")
    if not raw_client_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Client ID"
        )

    try:
        client: ApplicationClient = await get_client_from_client_id(
            session=session,
            client_id=uuid.UUID(raw_client_id)
        )
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Client ID"
        )

    g_jwt: ExchangedGoogleCode = await exchange_google_code(code=code)
    g_user: VerifiedGoogleUser = verify_google_token(token=g_jwt.id_token)
    s_user: VerifiedApplicationUser = await get_or_create_application_user(
        first_name=g_user.first_name,
        last_name=g_user.last_name,
        email=g_user.email,
        client_id=client.id,
        login_provider="Google",
        external_id=g_user.sub,
        session=session
    )

    response = RedirectResponse(
        url=decrypt(client.encrypted_redirect_uri),
        status_code=status.HTTP_302_FOUND
    )

    session_id = await post_session(
        session=session,
        user_id=s_user.user_id
    )

    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=config.COOKIE_SECURE,
        samesite="none" if config.COOKIE_SECURE else "lax",
        max_age=config.COOKIE_MAX_AGE,
        path="/"
    )

    log.debug(f"Set cookie for User ID: '{s_user.user_id}' for Client: '{s_user.client_id}'")

    return response