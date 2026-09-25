from app.core.config import settings
from app.ml.data.loader import clean_transactions, load_raw_data
from app.ml.training.baseline import evaluate_popularity_baseline
from app.ml.training.evaluation import leave_one_out_split
from app.ml.training.metrics import evaluate_model


def main() -> None:
    df = load_raw_data(settings.data_path)
    df_clean = clean_transactions(df)

    train_df, test_df = leave_one_out_split(df_clean)

    print(f"Train interactions: {len(train_df)}")
    print(f"Test interactions: {len(test_df)}")

    for k in [5, 10]:
        popularity_precision = evaluate_popularity_baseline(
            train_df=train_df,
            test_df=test_df,
            k=k,
        )

        cosine_precision = evaluate_model(
            train_df=train_df,
            test_df=test_df,
            k=k,
        )

        binary_cosine_precision = evaluate_model(
            train_df=train_df,
            test_df=test_df,
            k=k,
            binary=True,
        )

        print(f"\nPrecision@{k}")
        print(
            f"Popularity baseline: "
            f"{popularity_precision:.4f}"
        )
        print(
            f"Cosine recommender: "
            f"{cosine_precision:.4f}"
        )
        print(
            f"Cosine recommender (binary): "
            f"{binary_cosine_precision:.4f}"
        )


if __name__ == "__main__":
    main()