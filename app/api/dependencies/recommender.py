from functools import lru_cache

from app.core.config import settings
from app.ml.inference.recommender import ProductRecommender


@lru_cache
def get_recommender() -> ProductRecommender:
    return ProductRecommender.load(
        path=settings.model_path,
    )