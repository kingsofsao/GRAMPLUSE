"""
preprocessing.py
------------------
Cleans rural employment-demand data:
  - normalizes dates into a consistent format
  - standardizes village name variants (case/spacing) into one canonical ID
  - removes exact duplicate records
  - handles missing demand values (interpolation, never blind zero-fill)
  - builds a Village x Week weekly time series

This is the step the poster refers to as "Cleaning & Standardization".
"""

import re
import pandas as pd
import numpy as np
from config import program_for_date


def _parse_date_cell(value):
    """Parse a single date value, trying explicit known formats first
    (fast path, unambiguous) before falling back to dateutil for anything
    else. Doing this per-cell -- rather than a single pd.to_datetime call
    over the whole column -- avoids pandas' mixed-format inference
    silently failing when a column has more than one date format in it
    (e.g. '2026-07-01' next to '01/07/26'), which is common in real
    MIS exports merged from multiple sources."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y", "%B %d %Y", "%b %d %Y", "%d-%m-%Y"):
        try:
            return pd.to_datetime(text, format=fmt)
        except (ValueError, TypeError):
            continue
    # last resort: let dateutil guess, day-first (Indian date convention)
    try:
        return pd.to_datetime(text, dayfirst=True)
    except (ValueError, TypeError):
        return pd.NaT


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["date"].apply(_parse_date_cell)
    n_bad = df["date"].isna().sum()
    if n_bad:
        print(f"[preprocessing] Warning: {n_bad} rows had unparseable dates and were dropped.")
    return df.dropna(subset=["date"])


def standardize_village_name(name: str) -> str:
    """Collapse case/spacing variants of the same village name into one
    canonical form. Real deployments should replace this with a proper
    geographical master-ID lookup (e.g. LGD codes) instead of string
    matching, since string matching can't distinguish two genuinely
    different villages that happen to share a similar name."""
    if not isinstance(name, str):
        return name
    cleaned = re.sub(r"\s+", " ", name.strip())
    return cleaned.title()


def standardize_geography(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["state", "district", "block", "panchayat"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    df["village"] = df["village"].apply(standardize_village_name)
    df["village_id"] = (
        df["state"] + "|" + df["district"] + "|" + df["block"] + "|" + df["village"]
    )
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["village_id", "date"], keep="first")
    removed = before - len(df)
    if removed:
        print(f"[preprocessing] Removed {removed} duplicate records.")
    return df


def handle_missing_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing demand values per village along the time axis
    rather than blindly zero-filling (zero-fill would wrongly look like a
    demand crash and could distort the momentum score)."""
    df = df.sort_values(["village_id", "date"]).copy()
    n_missing = df["demand"].isna().sum()
    if n_missing:
        df["demand"] = df.groupby("village_id")["demand"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        still_missing = df["demand"].isna().sum()
        print(f"[preprocessing] Interpolated {n_missing - still_missing} missing demand values"
              f"{f'; {still_missing} left as missing (insufficient history)' if still_missing else ''}.")
    return df


def build_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per village per ISO week, summing demand if a
    village has more than one raw record within the same week."""
    df = df.copy()
    iso = df["date"].dt.isocalendar()
    df["year"] = iso["year"]
    df["week"] = iso["week"]

    weekly = (
        df.groupby(
            ["state", "district", "block", "panchayat", "village", "village_id", "year", "week"],
            as_index=False,
        )
        .agg(demand=("demand", "sum"), date=("date", "min"))
    )
    weekly = weekly.sort_values(["village_id", "year", "week"]).reset_index(drop=True)
    weekly["program"] = weekly["date"].apply(program_for_date)
    return weekly


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(df)
    df = standardize_geography(df)
    df = deduplicate(df)
    df = handle_missing_demand(df)
    weekly = build_weekly_series(df)
    return weekly


if __name__ == "__main__":
    from data_loader import load_raw_csv

    raw = load_raw_csv("data/raw/vb_gram_g_raw.csv")
    weekly = clean_pipeline(raw)
    weekly.to_csv("data/processed/village_weekly.csv", index=False)
    print(f"\nWrote {len(weekly)} weekly rows to data/processed/village_weekly.csv")
    print(weekly.head())
