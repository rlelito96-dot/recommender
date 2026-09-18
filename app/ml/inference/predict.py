import pandas as pd


def get_similar_products(
        product_id: str,
        similarity_df: pd.DataFrame,
        n: int = 5,
) -> pd.Series:
    if product_id not in similarity_df.columns:
        raise ValueError(f"Product {product_id} not found in similarity matrix")

    sorted_similarities = similarity_df[product_id].sort_values(ascending=False)
    return sorted_similarities.iloc[1 : n + 1]