from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies.database import get_db
from app.api.dependencies.recommender import get_recommender
from app.application.use_cases.recommend_products import (
    RecommendProductsUseCase,
)
from app.infrastructure.repositories.product_interaction import (
    ProductInteractionRepository,
)
from app.ml.inference.recommender import ProductRecommender
from app.api.schemas.recommendations import RecommendationResponse


router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)


@router.get(
    "/{customer_id}",
    response_model=RecommendationResponse,
)
def get_recommendations(
    customer_id: int,
    n: int = 5,
    db: Session = Depends(get_db),
    recommender: ProductRecommender = Depends(get_recommender),
) -> RecommendationResponse:
    repository = ProductInteractionRepository(session=db)

    use_case = RecommendProductsUseCase(
        repository=repository,
        recommender=recommender,
    )

    recommendations = use_case.execute(
        customer_id=customer_id,
        n=n,
    )

    return RecommendationResponse(
        customer_id=customer_id,
        recommendations=recommendations,
    )