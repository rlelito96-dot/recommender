from pathlib import Path

import pandas as pd
import pytest

from app.ml.data.loader import clean_transactions, load_raw_data


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["536365", "536365", "536366", "536367"],
            "StockCode": ["10002", "10080", "10109", "10120"],
            "Description": ["ITEM A", "ITEM B", "ITEM C", "ITEM D"],
            "Quantity": [6, -2, 8, 0],
            "InvoiceDate": pd.to_datetime(["2010-12-01"] * 4),
            "Price": [2.55, 3.39, 7.65, 0.0],
            "Customer ID": [17850.0, 17850.0, None, 13047.0],
            "Country": ["United Kingdom"] * 4,
        }
    )


class TestLoadRawData:
    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        fake_path = tmp_path / "nonexistent.xlsx"
        with pytest.raises(FileNotFoundError):
            load_raw_data(fake_path)


class TestCleanTransactions:
    def test_removes_rows_without_customer_id(
        self, sample_raw_df: pd.DataFrame
    ) -> None:
        cleaned = clean_transactions(sample_raw_df)
        assert cleaned["Customer ID"].isna().sum() == 0

    def test_removes_negative_quantity(self, sample_raw_df: pd.DataFrame) -> None:
        cleaned = clean_transactions(sample_raw_df)
        assert (cleaned["Quantity"] > 0).all()

    def test_removes_zero_price(self, sample_raw_df: pd.DataFrame) -> None:
        cleaned = clean_transactions(sample_raw_df)
        assert (cleaned["Price"] > 0).all()

    def test_customer_id_is_int(self, sample_raw_df: pd.DataFrame) -> None:
        cleaned = clean_transactions(sample_raw_df)
        assert cleaned["Customer ID"].dtype == int

    def test_result_row_count(self, sample_raw_df: pd.DataFrame) -> None:
        cleaned = clean_transactions(sample_raw_df)
        assert len(cleaned) == 1
