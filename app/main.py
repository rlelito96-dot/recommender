from fastapi import FastAPI

from app.api.exception_handlers import customer_not_found_handler
from app.api.routers.recommendations import router as recommendations_router
from app.core.exceptions import CustomerNotFoundException


app = FastAPI(
    title="Product Recommender API",
    version="1.0.0",
)

app.add_exception_handler(
    CustomerNotFoundException,
    customer_not_found_handler,
)

app.include_router(recommendations_router)