# Import every model so it registers on Base.metadata.
# Any import of database.schemas.* (e.g. Base in create_tables) runs this first.
from .agent_config import AgentConfigT
from .agent_sharing import AgentSharingT
from .base import Base
from .clients import ClientsT
from .documents_user import DocumentsUsersT
from .mcp import MCPsT
from .message_threads import MessageThreadsT
from .model_sampling import ModelSamplingT
from .models import ModelsT
from .persons_users import PersonsT, UsersT
from .providers import ProvidersT
from .user_keys import UserKeysT
from .user_sessions import SessionsT
from .user_skills import UserSkillsT
from .vector_store import VectorStoreSettingsT
from .vector_store_collections import VectorStoreCollectionT

__all__ = [
    "AgentConfigT",
    "AgentSharingT",
    "Base",
    "ClientsT",
    "DocumentsUsersT",
    "MCPsT",
    "MessageThreadsT",
    "ModelSamplingT",
    "ModelsT",
    "PersonsT",
    "ProvidersT",
    "SessionsT",
    "UserKeysT",
    "UserSkillsT",
    "VectorStoreSettingsT",
    "VectorStoreCollectionT",
    "UsersT",
]
