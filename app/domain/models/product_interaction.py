from dataclasses import dataclass


@dataclass
class ProductInteraction:
    customer_id: int
    product_id: str
    quantity: int
    price: float