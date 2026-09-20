import pandas as pd

from app.ml.training.evaluation import leave_one_out_split


def test_leave_one_out_uses_latest_product_interaction() -> None:
    df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 1, 1],
            "StockCode": ["A", "A", "B", "C"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-05-01",
                    "2010-03-01",
                    "2010-04-01",
                ]
            ),
            "Quantity": [1, 1, 1, 1],
        }
    )

    train_df, test_df = leave_one_out_split(df)

    assert len(test_df) == 1
    assert test_df.iloc[0]["StockCode"] == "A"
    assert set(train_df["StockCode"]) == {"B", "C"}