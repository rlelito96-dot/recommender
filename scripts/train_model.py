from pathlib import Path

from app.core.config import settings
from app.ml.data.loader import clean_transactions, load_raw_data
from app.ml.features.build_features import (
    build_customer_product_matrix,
    compute_product_similarity,
)
from app.ml.inference.recommender import ProductRecommender


def main() -> None:
    df = load_raw_data(settings.data_path)
    df_clean = clean_transactions(df)

    customer_product_matrix = build_customer_product_matrix(
        df_clean,
        binary=True,
    )

    similarity_df = compute_product_similarity(
        customer_product_matrix,
    )

    recommender = ProductRecommender(
        similarity_df=similarity_df,
    )

    model_path = Path(settings.model_path)

    recommender.save(
        path=model_path,
    )

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()