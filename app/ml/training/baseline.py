import pandas as pd

from app.ml.training.metrics import precision_at_k



def get_popular_products(
    train_df: pd.DataFrame,
    n: int = 5,
) -> list[str]:

    popular_products = (
        train_df.groupby("StockCode")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )

    return popular_products.index.tolist()

def evaluate_popularity_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
) -> float:

    popular_products = get_popular_products(train_df, n=k)

    scores = []

    for _, row in test_df.iterrows():
        score = precision_at_k(
            recommendations=popular_products,
            target=row["StockCode"],
            k=k,
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)



