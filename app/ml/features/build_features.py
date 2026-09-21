import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_customer_product_matrix(
    df_clean: pd.DataFrame,
    binary: bool = False,
) -> pd.DataFrame:
    matrix = df_clean.pivot_table(
        index="Customer ID",
        columns="StockCode",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )

    if binary:
        matrix = (matrix > 0).astype(int)

    return matrix


def compute_product_similarity(customer_product_matrix: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(customer_product_matrix.T)
    return pd.DataFrame(
        similarity,
        index=customer_product_matrix.columns,
        columns=customer_product_matrix.columns,
    )
