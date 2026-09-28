from unittest.mock import Mock

from app.application.use_cases.recommend_products import (
    RecommendProductsUseCase,
)


def test_recommend_products() -> None:
    repository = Mock()
    repository.get_customer_products.return_value = [
        "A",
        "B",
    ]

    recommender = Mock()
    recommender.recommend_for_history.return_value = [
        "C",
        "D",
        "E",
    ]

    use_case = RecommendProductsUseCase(
        repository=repository,
        recommender=recommender,
    )

    recommendations = use_case.execute(
        customer_id=123,
        n=3,
    )

    assert recommendations == ["C", "D", "E"]

    repository.get_customer_products.assert_called_once_with(
        customer_id=123,
    )

    recommender.recommend_for_history.assert_called_once_with(
        customer_products=["A", "B"],
        n=3,
    )