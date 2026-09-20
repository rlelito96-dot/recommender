import pandas as pd


def leave_one_out_split(
    df: pd.DataFrame,
    min_interactions: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test using leave-one-out."""

    interactions = (
        df.groupby(["Customer ID", "StockCode"], as_index=False)["InvoiceDate"]
        .max()
        .sort_values(["Customer ID", "InvoiceDate"])
    )

    test_rows = []

    for customer_id, customer_df in interactions.groupby("Customer ID"):
        if len(customer_df) < min_interactions:
            continue

        test_row = customer_df.iloc[-1]
        test_rows.append(test_row)

    test_df = pd.DataFrame(test_rows)

    test_pairs = test_df.set_index(["Customer ID", "StockCode"]).index

    train_df = df.set_index(["Customer ID", "StockCode"])
    train_df = train_df.drop(index=test_pairs, errors="ignore")
    train_df = train_df.reset_index()

    return train_df, test_df.reset_index(drop=True)