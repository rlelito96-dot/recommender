from app.infrastructure.repositories.product_interaction import (
    ProductInteractionRepository,
)
from app.ml.inference.recommender import ProductRecommender


class RecommendProductsUseCase:
    def __init__(
        self,
        repository: ProductInteractionRepository,
        recommender: ProductRecommender,
    ) -> None:
        self.repository = repository
        self.recommender = recommender

    def execute(
        self,
        customer_id: int,
        n: int = 5,
    ) -> list[str]:
        customer_products = self.repository.get_customer_products(
            customer_id=customer_id,
        )

        return self.recommender.recommend_for_history(
            customer_products=customer_products,
            n=n,
        )