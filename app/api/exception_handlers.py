from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import CustomerNotFoundException


def customer_not_found_handler(
    request: Request,
    exc: CustomerNotFoundException,
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
    )