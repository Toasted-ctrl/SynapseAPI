from fastapi import APIRouter

from .models import StatusRead

router = APIRouter()


@router.get("/status", tags=["Status"], response_model=StatusRead)
def get_status():
    return {"status": "OK"}
