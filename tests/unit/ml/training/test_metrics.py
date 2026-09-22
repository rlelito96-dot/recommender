import pandas as pd

from app.ml.training.metrics import precision_at_k, evaluate_model
from app.ml.training.metrics import (
    evaluate_model,
    filter_seen_products,
    precision_at_k,
)


def test_precision_at_k_returns_one_for_hit() -> None:
    recommendations = ["A", "B", "C", "D", "E"]

    result = precision_at_k(
        recommendations=recommendations,
        target="C",
        k=5,
    )

    assert result == 1.0


def test_precision_at_k_returns_zero_for_miss() -> None:
    recommendations = ["A", "B", "C", "D", "E"]

    result = precision_at_k(
        recommendations=recommendations,
        target="F",
        k=5,
    )

    assert result == 0.0


def test_precision_at_k_only_considers_top_k() -> None:
    recommendations = ["A", "B", "C", "D", "E"]

    result = precision_at_k(
        recommendations=recommendations,
        target="E",
        k=3,
    )

    assert result == 0.0


def test_evaluate_model_returns_score() -> None:
    train_df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, 2],
            "StockCode": ["A", "B", "A", "C"],
            "Quantity": [2, 1, 1, 2],
        }
    )

    test_df = pd.DataFrame(
        {
            "Customer ID": [1],
            "StockCode": ["C"],
            "InvoiceDate": pd.to_datetime(["2010-03-01"]),
        }
    )

    result = evaluate_model(
        train_df=train_df,
        test_df=test_df,
        k=2,
    )

    assert 0.0 <= result <= 1.0

def test_filter_seen_products():
    recommendations = ["A", "B", "C", "D"]
    seen_products = {"A", "C"}

    result = filter_seen_products(
        recommendations=recommendations,
        seen_products=seen_products,
    )

    assert result == ["B", "D"]