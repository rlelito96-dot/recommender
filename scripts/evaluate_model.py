from app.core.config import settings
from app.ml.data.loader import clean_transactions, load_raw_data
from app.ml.training.evaluation import leave_one_out_split
from app.ml.training.metrics import evaluate_model


def main() -> None:
    df = load_raw_data(settings.data_path)
    df_clean = clean_transactions(df)

    train_df, test_df = leave_one_out_split(df_clean)

    precision_at_5 = evaluate_model(
        train_df=train_df,
        test_df=test_df,
        k=5,
    )

    print(f"Train interactions: {len(train_df)}")
    print(f"Test interactions: {len(test_df)}")
    print(f"Precision@5: {precision_at_5:.4f}")


if __name__ == "__main__":
    main()