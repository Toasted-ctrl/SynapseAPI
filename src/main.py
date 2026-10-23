import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis

from api import include_routers
from core.config import config
from core.logging import get_logger
from database.session import async_engine
from exch import register_exception_handlers


log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = aioredis.from_url(
        f"redis://{config.REDIS_USER}:{config.REDIS_PASSWORD}@{config.REDIS_HOSTNAME}:{config.REDIS_PORT}/0",
    )
    FastAPICache.init(RedisBackend(redis=redis), prefix=config.REDIS_PREFIX)
    log.info("Redis cache initialized")
    yield
    await redis.close()
    await async_engine.dispose()

app = FastAPI(
    title=config.APP_NAME,
    description=(
        "SynapseAPI provides a unified interface for AI model interactions, "
        "including chat, agents, embeddings, translations, and document management. "
        "It supports multiple providers (OpenAI, Anthropic, Ollama, etc.) and offers "
        "features like vector storage, MCP tool integration, and user authentication."
    ),
    version=config.APP_VERSION,
    contact={
        "name": config.APP_MAINTAINER,
    },
    lifespan=lifespan
)

register_exception_handlers(app=app)

include_routers(app=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def secure_logging(request: Request, call_next):
    response = await call_next(request)

    path = request.url.path
    if "/auth/" in path or "/callback" in path:
        log.info(
            f"{request.client.host} - "
            f"\"{request.method} {path}\" "
            f"{response.status_code}"
        )
    else:
        log.info(
            f"{request.client.host} - "
            f"\"{request.method} {request.url}\" "
            f"{response.status_code}"
        )

    return response

if __name__ == "__main__":

    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000,
        access_log=False
    )