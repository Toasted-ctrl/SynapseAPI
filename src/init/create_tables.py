from sqlalchemy import create_engine

from core.config import config
from core.logging import get_logger
from database.schemas.base import Base
from database.schemas.agent_config import AgentConfigT
from database.schemas.clients import ClientsT
from database.schemas.documents_user import DocumentsUsersT
from database.schemas.mcp import MCPsT
from database.schemas.message_threads import MessageThreadsT
from database.schemas.model_sampling import ModelSamplingT
from database.schemas.models import ModelsT
from database.schemas.persons_users import PersonsT, UsersT
from database.schemas.providers import ProvidersT
from database.schemas.user_keys import UserKeysT
from database.schemas.user_sessions import SessionsT
from database.schemas.user_skills import UserSkillsT
from database.schemas.vector_store import VectorStoreSettingsT
from database.schemas.vector_store_collections import VectorStoreCollectionT


log = get_logger()


def create_tables():

    try:
        engine = create_engine(url=config.PG_DB_URL)
        log.info("Created engine to create required tables...")

        Base.metadata.create_all(bind=engine)
        log.info("Created required tables...")

        return

    except Exception as e:
        log.error(f"Unexpected error: {e}. Shutting down...")
        raise SystemExit(1)

    finally:
        if engine:
            engine.dispose()
            log.info("Disposed engine...")