import json
from functools import cached_property
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar, Set


BASE_DIR = Path(__file__).resolve().parent.parent.parent
_env_file = BASE_DIR / ".env"
    

class Config(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_env_file if _env_file.exists() else None,
        extra="ignore"
    )

    _SKIP_EMPTY_CHECK: ClassVar[Set[str]] = {
        "_SKIP_EMPTY_CHECK",
        "APP_NAME",
        "APP_VERSION",
        "APP_MAINTAINER",
        "_SKIP_GOOGLE_ENV_VARS",
    }

    APP_NAME: str = "SynapseAPI"
    APP_MAINTAINER: str = "Toasted-ctrl"
    APP_VERSION: str = ""

    REDIS_USER: str = ""
    REDIS_HOSTNAME: str = ""
    REDIS_PASSWORD: str = ""
    REDIS_PREFIX: str = ""
    REDIS_PORT: int

    PG_HOSTNAME: str = ""
    PG_DATABASE: str = ""
    PG_USERNAME: str = ""
    PG_PASSWORD: str = ""
    PG_DIALECT: str = ""
    PG_DRIVER: str = ""
    PG_PORT: int

    ENCRYPTION_KEY: str = ""

    BLIND_INDEX_KEY: str = ""

    JWT_SECRET: str = ""

    LOG_LEVEL: str = ""

    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_BASE_URL: str = ""

    ALLOWED_ORIGINS: str = ""

    COOKIE_SECURE: bool = True
    COOKIE_MAX_AGE: int

    ENABLE_GOOGLE_LOGIN: bool = False
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_REDIRECT_URI: str = ""
    GOOGLE_AUTH_URL: str = ""
    GOOGLE_HMAC: str = ""
    GOOGLE_TOKEN_URL: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    WEB_SEARCH_URL: str = ""
    WEB_SEARCH_TRANSPORT: str = ""

    _SKIP_GOOGLE_ENV_VARS: ClassVar[set[str]] = {
        "GOOGLE_CLIENT_ID",
        "GOOGLE_REDIRECT_URI",
        "GOOGLE_AUTH_URL",
        "GOOGLE_HMAC",
        "GOOGLE_TOKEN_URL",
        "GOOGLE_CLIENT_SECRET"
    }


    def model_post_init(self, context) -> None:

        # Continue with settings check.
        all_fields = set(self.__class__.__annotations__.keys())

        unpopulated = all_fields - self._SKIP_EMPTY_CHECK
        unset = all_fields - self.model_fields_set - self._SKIP_EMPTY_CHECK

        if not self.ENABLE_GOOGLE_LOGIN:
            print(f"Google Login disabled. Skipping environment variables: {' ,'.join(sorted(self._SKIP_GOOGLE_ENV_VARS))} ...")
            unset - self._SKIP_GOOGLE_ENV_VARS
            unpopulated - self._SKIP_GOOGLE_ENV_VARS
        
        empty = {
            name
            for name in (unpopulated)
            if isinstance(getattr(self, name, None), str)
            and not getattr(self, name).strip()
        }
        problems = unset | empty
        if problems:
            msgs = []
            if unset:
                msgs.append(f"Missing: {', '.join(sorted(unset))}")
            if empty:
                msgs.append(f"Empty: {', '.join(sorted(empty))}")
            print(
                "The following fields have problems in the .env:\n"
                + "\n".join(msgs)
                + "\nShutting down"
            )
            raise SystemExit(1)
        print("Environment variables loaded")


    @cached_property
    def GOOGLE_HMAC_SECRET(self) -> bytes:
        """Returns Google HMAC secret"""
        return self.GOOGLE_HMAC.encode('utf-8')


    @cached_property
    def BLIND_INDEX_HMAC_KEY(self) -> bytes:
        """Returns blind index key."""
        return self.BLIND_INDEX_KEY.encode('utf-8')


    @cached_property
    def CORS_ALLOWED_ORIGINS(self) -> list[str]:
        """Returns a list of allowed origins."""
        return json.loads(self.ALLOWED_ORIGINS)


    @cached_property
    def PG_DB_URL(self) -> str:
        """Returns the database URL."""
        return (
            f"{self.PG_DIALECT}+{self.PG_DRIVER}://"
            f"{self.PG_USERNAME}:{self.PG_PASSWORD}@"
            f"{self.PG_HOSTNAME}:{self.PG_PORT}/{self.PG_DATABASE}"
        )


    @cached_property
    def ASYNC_PG_DB_URL(self) -> str:
        """Returns the database URL for the PG connection pool."""
        return (
            f"{self.PG_DIALECT}+asyncpg://"
            f"{self.PG_USERNAME}:{self.PG_PASSWORD}@"
            f"{self.PG_HOSTNAME}:{self.PG_PORT}/{self.PG_DATABASE}"
        )


    @cached_property
    def PG_CHECKPOINTER_URL(self) -> str:
        """Returns checkpointer database URL."""
        return (
            f"{self.PG_DIALECT}://"
            f"{self.PG_USERNAME}:{self.PG_PASSWORD}@"
            f"{self.PG_HOSTNAME}/{self.PG_DATABASE}"
        )
    

config = Config()