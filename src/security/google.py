from dataclasses import dataclass

import httpx
from fastapi import HTTPException, status
from google.auth.transport import requests
from google.oauth2 import id_token as google_id_token

from core.config import config
from core.logging import get_logger

log = get_logger()


@dataclass(frozen=True)
class ExchangedGoogleCode:
    id_token: str
    refresh_token: str
    scope: str
    token_type: str
    expires_in: int
    access_token: str | None = (
        None  # Google only sends the refresh token on the FIRST authorization.
    )


async def exchange_google_code(code: str) -> ExchangedGoogleCode:
    async with httpx.AsyncClient() as client:
        log.debug("Exchanging code with Google...")
        response = await client.post(
            config.GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": config.GOOGLE_CLIENT_ID,
                "client_secret": config.GOOGLE_CLIENT_SECRET,
                "redirect_uri": config.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )

        if response.status_code != 200:
            log.warning("Failed to exchange code with Google")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to exchange code with Google",
            )

        log.debug("Exchanged code with Google...")
        r_data = response.json()
        return ExchangedGoogleCode(
            id_token=r_data.get("id_token"),
            refresh_token=r_data.get("refresh_token"),
            scope=r_data.get("scope"),
            token_type=r_data.get("token_type"),
            expires_in=r_data.get("expires_in"),
            access_token=r_data.get("access_token", None),
        )


@dataclass(frozen=True)
class VerifiedGoogleUser:
    iss: str
    sub: str
    email: str
    email_verified: bool
    first_name: str
    last_name: str


def verify_google_token(token: str) -> VerifiedGoogleUser:
    try:
        log.debug(f"Attempting to verify Google ID token: '{token[:20]}'...")
        user_info = google_id_token.verify_oauth2_token(
            token, requests.Request(), audience=config.GOOGLE_CLIENT_ID
        )
        log.debug("Verified Google ID token...")
        return VerifiedGoogleUser(
            iss=user_info.get("iss"),
            sub=user_info.get("sub"),
            email=user_info.get("email"),
            email_verified=user_info.get("email_verified"),
            first_name=user_info.get("given_name"),
            last_name=user_info.get("family_name"),
        )

    except ValueError:
        log.warning(f"Invalid Google ID token: '{token[:20]}'...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google ID token"
        )
