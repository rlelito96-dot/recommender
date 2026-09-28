from fastapi import FastAPI

from app.api.routers.recommendations import router as recommendations_router


app = FastAPI(
    title="Product Recommender API",
    version="1.0.0",
)

app.include_router(recommendations_router)