import pandas as pd

from app.core.config import settings
from app.infrastructure.db.models import ProductInteractionModel
from app.infrastructure.db.session import SessionLocal
from app.ml.data.loader import clean_transactions, load_raw_data


def main() -> None:
    print("Loading raw data...")

    df = load_raw_data(settings.data_path)
    df_clean = clean_transactions(df)

    print(f"Rows to import: {len(df_clean)}")

    session = SessionLocal()

    try:
        interactions = [
            ProductInteractionModel(
                customer_id=int(row["Customer ID"]),
                product_id=str(row["StockCode"]),
                quantity=int(row["Quantity"]),
                price=float(row["Price"]),
            )
            for _, row in df_clean.iterrows()
        ]

        session.add_all(interactions)
        session.commit()

        print(f"Imported {len(interactions)} interactions.")

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()