import pandas as pd
import pytest

from app.ml.inference.predict import get_similar_products


@pytest.fixture
def similarity_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 0.9, 0.1],
            "B": [0.9, 1.0, 0.2],
            "C": [0.1, 0.2, 1.0],
        },
        index=["A", "B", "C"],
    )


class TestGetSimilarProducts:
    def test_excludes_self_from_results(self, similarity_df: pd.DataFrame) -> None:
        result = get_similar_products("A", similarity_df, n=2)
        assert "A" not in result.index

    def test_returns_correct_number_of_results(self, similarity_df: pd.DataFrame) -> None:
        result = get_similar_products("A", similarity_df, n=2)
        assert len(result) == 2

    def test_results_sorted_descending(self, similarity_df: pd.DataFrame) -> None:
        result = get_similar_products("A", similarity_df, n=2)
        assert result.iloc[0] >= result.iloc[1]

    def test_most_similar_product_is_first(self, similarity_df: pd.DataFrame) -> None:
        result = get_similar_products("A", similarity_df, n=2)
        assert result.index[0] == "B"

    def test_raises_for_unknown_product(self, similarity_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_similar_products("UNKNOWN", similarity_df, n=2)