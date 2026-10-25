import openai
from fastapi import Request, status
from fastapi.responses import JSONResponse


async def not_found_handler(request: Request, exc: openai.NotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Model not found: {exc.message}"},
    )
