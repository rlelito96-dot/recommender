from __future__ import annotations

import pandas as pd

from app.ml.inference.predict import get_similar_products


class ProductRecommender:

    def __init__(self, similarity_df: pd.DataFrame) -> None:
        self.similarity_df = similarity_df

    def recommend_for_history(
        self,
        customer_products: list[str],
        n: int = 5,
    ) -> list[str]:

        candidate_scores: dict[str, float] = {}

        seen_products = set(customer_products)

        for product_id in customer_products:
            if product_id not in self.similarity_df.columns:
                continue

            recommendations = get_similar_products(
                product_id=product_id,
                similarity_df=self.similarity_df,
                n=20,
            )

            for recommended_product, similarity_score in recommendations.items():
                candidate_scores[recommended_product] = (
                    candidate_scores.get(recommended_product, 0.0)
                    + similarity_score
                )

        recommendations = [
            product_id
            for product_id, _ in sorted(
                candidate_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        recommendations = [
            product_id
            for product_id in recommendations
            if product_id not in seen_products
        ]

        return recommendations[:n]