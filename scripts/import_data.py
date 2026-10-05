import pandas as pd

from app.core.config import settings
from app.infrastructure.db.models import ProductInteractionModel
from app.infrastructure.db.session import SessionLocal
from app.ml.data.loader import clean_transactions, load_raw_data


BATCH_SIZE = 5_000


def main() -> None:
    print("Loading raw data...")

    df = load_raw_data(settings.data_path)
    df_clean = clean_transactions(df)

    print(f"Rows to import: {len(df_clean)}")

    session = SessionLocal()

    try:
        for start in range(0, len(df_clean), BATCH_SIZE):
            batch = df_clean.iloc[start : start + BATCH_SIZE]

            interactions = [
                {
                    "customer_id": int(row["Customer ID"]),
                    "product_id": str(row["StockCode"]),
                    "quantity": int(row["Quantity"]),
                    "price": float(row["Price"]),
                }
                for _, row in batch.iterrows()
            ]

            session.execute(
                ProductInteractionModel.__table__.insert(),
                interactions,
            )

            session.commit()

            print(
                f"Imported {min(start + BATCH_SIZE, len(df_clean))}"
                f"/{len(df_clean)}"
            )

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()