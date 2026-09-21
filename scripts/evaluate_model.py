from app.core.config import settings
from app.ml.data.loader import clean_transactions, load_raw_data
from app.ml.training.evaluation import leave_one_out_split
from app.ml.training.metrics import evaluate_model
from app.ml.training.baseline import evaluate_popularity_baseline


def main() -> None:
    df = load_raw_data(settings.data_path)
    df_clean = clean_transactions(df)

    train_df, test_df = leave_one_out_split(df_clean)

    popularity_precision_at_5 = evaluate_popularity_baseline(
        train_df=train_df,
        test_df=test_df,
        k=5,
    )

    cosine_precision_at_5 = evaluate_model(
        train_df=train_df,
        test_df=test_df,
        k=5,
    )

    binary_cosine_precision_at_5 = evaluate_model(
        train_df=train_df,
        test_df=test_df,
        k=5,
        binary=True,
    )

    print(f"Train interactions: {len(train_df)}")
    print(f"Test interactions: {len(test_df)}")
    print(
        f"Popularity baseline Precision@5: "
        f"{popularity_precision_at_5:.4f}"
    )
    print(f"Cosine recommender Precision@5: {cosine_precision_at_5:.4f}")
    print(
        f"Cosine recommender (binary) Precision@5: "
        f"{binary_cosine_precision_at_5:.4f}"
    )


if __name__ == "__main__":
    main()