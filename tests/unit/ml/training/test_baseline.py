import pandas as pd

from app.ml.training.baseline import get_popular_products
from app.ml.training.baseline import evaluate_popularity_baseline


def test_get_popular_products_returns_products_sorted_by_quantity():
    df = pd.DataFrame(
        {
            "StockCode": ["A", "B", "C", "A", "B"],
            "Quantity": [10, 5, 20, 5, 10],
        }
    )

    result = get_popular_products(df, n=3)

    assert result == ["C", "A", "B"]

def test_evaluate_popularity_baseline_returns_precision():
    train_df = pd.DataFrame(
        {
            "StockCode": ["A", "B", "C", "A", "B"],
            "Quantity": [10, 5, 20, 5, 10],
        }
    )

    test_df = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "StockCode": ["C", "X"],
        }
    )

    result = evaluate_popularity_baseline(
        train_df=train_df,
        test_df=test_df,
        k=3,
    )

    assert result == 0.5