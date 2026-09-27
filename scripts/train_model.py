from pathlib import Path

import mlflow

from app.core.config import settings
from app.ml.data.loader import clean_transactions, load_raw_data
from app.ml.features.build_features import (
    build_customer_product_matrix,
    compute_product_similarity,
)
from app.ml.inference.recommender import ProductRecommender
from app.ml.training.evaluation import leave_one_out_split
from app.ml.training.metrics import evaluate_model


def main() -> None:
    df = load_raw_data(settings.data_path)
    df_clean = clean_transactions(df)

    train_df, test_df = leave_one_out_split(df_clean)

    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_type", "product_product_cosine")
        mlflow.log_param("binary", True)

        customer_product_matrix = build_customer_product_matrix(
            train_df,
            binary=True,
        )

        similarity_df = compute_product_similarity(
            customer_product_matrix,
        )

        recommender = ProductRecommender(
            similarity_df=similarity_df,
        )

        precision_at_5 = evaluate_model(
            train_df=train_df,
            test_df=test_df,
            k=5,
            binary=True,
        )

        precision_at_10 = evaluate_model(
            train_df=train_df,
            test_df=test_df,
            k=10,
            binary=True,
        )

        mlflow.log_metric("precision_at_5", precision_at_5)
        mlflow.log_metric("precision_at_10", precision_at_10)

        model_path = Path(settings.model_path)

        recommender.save(
            path=model_path,
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        print(f"Model saved to: {model_path}")
        print(f"Precision@5: {precision_at_5:.4f}")
        print(f"Precision@10: {precision_at_10:.4f}")


if __name__ == "__main__":
    main()