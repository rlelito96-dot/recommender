import numpy as np
import pandas as pd

from app.ml.features.build_features import (
    build_customer_product_matrix,
    compute_product_similarity,
)


class TestBuildCustomerProductMatrix:
    def test_matrix_shape(self) -> None:
        df = pd.DataFrame(
            {
                "Customer ID": [1, 1, 2],
                "StockCode": ["A", "B", "A"],
                "Quantity": [2, 3, 1],
            }
        )
        matrix = build_customer_product_matrix(df)
        assert matrix.shape == (2, 2)

    def test_missing_combination_is_zero(self) -> None:
        df = pd.DataFrame(
            {
                "Customer ID": [1, 1, 2],
                "StockCode": ["A", "B", "A"],
                "Quantity": [2, 3, 1],
            }
        )
        matrix = build_customer_product_matrix(df)
        assert matrix.loc[2, "B"] == 0

    def test_sums_repeated_purchases(self) -> None:
        df = pd.DataFrame(
            {
                "Customer ID": [1, 1],
                "StockCode": ["A", "A"],
                "Quantity": [2, 3],
            }
        )
        matrix = build_customer_product_matrix(df)
        assert matrix.loc[1, "A"] == 5


class TestComputeProductSimilarity:
    def test_diagonal_is_one(self) -> None:
        matrix = pd.DataFrame(
            {"A": [1, 0, 2], "B": [0, 1, 0]},
            index=[1, 2, 3],
        )
        similarity = compute_product_similarity(matrix)
        assert np.allclose(np.diag(similarity.values), 1.0)

    def test_matrix_is_symmetric(self) -> None:
        matrix = pd.DataFrame(
            {"A": [1, 0, 2], "B": [0, 1, 0], "C": [3, 1, 0]},
            index=[1, 2, 3],
        )
        similarity = compute_product_similarity(matrix)
        assert np.allclose(similarity.values, similarity.values.T)

    def test_result_shape_matches_product_count(self) -> None:
        matrix = pd.DataFrame(
            {"A": [1, 0], "B": [0, 1], "C": [2, 3]},
            index=[1, 2],
        )
        similarity = compute_product_similarity(matrix)
        assert similarity.shape == (3, 3)
