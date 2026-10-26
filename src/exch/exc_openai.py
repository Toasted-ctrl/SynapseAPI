import openai
from fastapi import Request, status
from fastapi.responses import JSONResponse


async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    # Starlette types handlers as taking any Exception; registered for openai.NotFoundError.
    message = exc.message if isinstance(exc, openai.NotFoundError) else str(exc)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Model not found: {message}"},
    )
