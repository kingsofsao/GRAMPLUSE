"""
data_loader.py
---------------
Loads raw MGNREGA-style CSV data into a DataFrame.

Expected minimum columns (case-insensitive, flexible order):
    state, district, block, panchayat, village, date, demand

Swap `load_raw_csv` for a scraper/API call against the MGNREGA MIS portal
when you have real data access; everything downstream only depends on
this function returning a DataFrame with those columns.
"""

import pandas as pd

REQUIRED_COLUMNS = ["state", "district", "block", "panchayat", "village", "date", "demand"]


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    return df[REQUIRED_COLUMNS].copy()


if __name__ == "__main__":
    df = load_raw_csv("data/raw/mgnrega_raw.csv")
    print(df.head())
    print(f"\n{len(df)} rows loaded")
