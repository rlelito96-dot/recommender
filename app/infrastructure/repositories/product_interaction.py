from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infrastructure.db.models import ProductInteractionModel


class ProductInteractionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_customer_products(
        self,
        customer_id: int,
    ) -> list[str]:
        statement = (
            select(ProductInteractionModel.product_id)
            .where(
                ProductInteractionModel.customer_id == customer_id,
            )
            .distinct()
        )

        result = self.session.execute(statement)

        return list(result.scalars().all())