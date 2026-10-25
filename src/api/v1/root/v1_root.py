from fastapi import APIRouter, Request, Response
from fastapi_cache.decorator import cache

from core.cache import cache_key_builder
from core.config import config

from .models import RootRead

router = APIRouter()


@router.get(
    "/root",
    response_model=RootRead,
    description=("Retrieves API name, version and contact/maintainer information."),
    tags=["Root"],
)
@cache(expire=300, key_builder=cache_key_builder)
def get_root(request: Request, response: Response) -> RootRead:

    return {
        "application_name": config.APP_NAME,
        "version": config.APP_VERSION,
        "contact": {"maintainer": config.APP_MAINTAINER},
    }
