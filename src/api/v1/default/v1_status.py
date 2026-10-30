from fastapi import APIRouter

from .models import StatusRead

router = APIRouter()


@router.get("/status", tags=["Default"], response_model=StatusRead)
def get_status():
    return {"status": "OK"}
