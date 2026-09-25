import pandas as pd

from app.ml.inference.recommender import ProductRecommender


def test_recommend_for_history_excludes_seen_products() -> None:
    similarity_df = pd.DataFrame(
        {
            "A": [1.0, 0.9, 0.8, 0.7],
            "B": [0.9, 1.0, 0.6, 0.5],
            "C": [0.8, 0.6, 1.0, 0.4],
            "D": [0.7, 0.5, 0.4, 1.0],
        },
        index=["A", "B", "C", "D"],
    )

    recommender = ProductRecommender(similarity_df)

    recommendations = recommender.recommend_for_history(
        customer_products=["A"],
        n=2,
    )

    assert recommendations == ["B", "C"]