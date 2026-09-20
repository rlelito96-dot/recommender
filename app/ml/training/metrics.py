import pandas as pd

from app.ml.features.build_features import (
    build_customer_product_matrix,
    compute_product_similarity,
)
from app.ml.inference.predict import get_similar_products


def precision_at_k(
    recommendations: list[str],
    target: str,
    k: int,
) -> float:
    """Calculate Precision@K for a single target item."""
    top_k = recommendations[:k]

    return 1.0 if target in top_k else 0.0


def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
) -> float:
    """Evaluate product recommendations using Precision@K."""

    customer_product_matrix = build_customer_product_matrix(train_df)
    similarity_df = compute_product_similarity(customer_product_matrix)

    scores = []

    for _, row in test_df.iterrows():
        target_product = row["StockCode"]

        customer_products = train_df[
            train_df["Customer ID"] == row["Customer ID"]
        ]["StockCode"].unique()

        for product_id in customer_products:
            if product_id == target_product:
                continue

            recommendations = get_similar_products(
                product_id=product_id,
                similarity_df=similarity_df,
                n=k,
            ).index.tolist()

            score = precision_at_k(
                recommendations=recommendations,
                target=target_product,
                k=k,
            )

            scores.append(score)
            break

    if not scores:
        return 0.0

    return sum(scores) / len(scores)