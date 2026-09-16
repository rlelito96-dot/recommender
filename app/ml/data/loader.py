from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Wczytuje surowy plik Excel z transakcjami."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku danych: {path}. "
            "Pobierz dataset z UCI i umiesc w data/raw/."
        )
    df = pd.read_excel(path, sheet_name=0)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna(subset=["Customer ID"])
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean = df_clean[df_clean["Price"] > 0]
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
    df_clean = df_clean.reset_index(drop=True)
    df_clean = df_clean.drop_duplicates()
    return df_clean
