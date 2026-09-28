from unittest.mock import Mock

from app.infrastructure.repositories.product_interaction import (
    ProductInteractionRepository,
)


def test_get_customer_products() -> None:
    session = Mock()

    result = Mock()
    result.scalars.return_value.all.return_value = [
        "A",
        "B",
        "C",
    ]

    session.execute.return_value = result

    repository = ProductInteractionRepository(session)

    products = repository.get_customer_products(customer_id=123)

    assert products == ["A", "B", "C"]
    session.execute.assert_called_once()