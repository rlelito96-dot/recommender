from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    customer_id: int
    recommendations: list[str]