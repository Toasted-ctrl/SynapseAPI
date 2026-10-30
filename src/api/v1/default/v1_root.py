from fastapi import APIRouter, Request, Response
from fastapi_cache.decorator import cache

from core.cache import cache_key_builder
from core.config import config

from .models import Contact, RootRead

router = APIRouter()


@router.get(
    "/root",
    response_model=RootRead,
    description=("Retrieves API name, version and contact/maintainer information."),
    tags=["Default"],
)
@cache(expire=300, key_builder=cache_key_builder)
async def get_root(request: Request, response: Response) -> RootRead:

    return RootRead(
        application_name=config.APP_NAME,
        version=config.APP_VERSION,
        contact=Contact(maintainer=config.APP_MAINTAINER),
    )
