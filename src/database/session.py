from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import config
from core.logging import get_logger

log = get_logger()


async_engine = create_async_engine(
    config.ASYNC_PG_DB_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that yields an async database session."""
    log.debug("Opening database session")
    async with AsyncSessionLocal() as session, session.begin():
        yield session
    log.debug("Closed database session")


@asynccontextmanager
async def get_db_session_ctx() -> AsyncGenerator[AsyncSession]:
    """For use outside of FastAPI (scripts, workers, etc.)."""
    log.debug("Opening database session")
    async with AsyncSessionLocal() as session, session.begin():
        yield session
    log.debug("Closed database session")
