import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_customer_product_matrix(df_clean: pd.DataFrame) -> pd.DataFrame:
    return df_clean.pivot_table(
        index="Customer ID",
        columns="StockCode",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )


def compute_product_similarity(customer_product_matrix: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(customer_product_matrix.T)
    return pd.DataFrame(
        similarity,
        index=customer_product_matrix.columns,
        columns=customer_product_matrix.columns,
    )
