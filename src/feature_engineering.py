"""
feature_engineering.py
------------------------
Builds the per-village-week feature set that everything downstream
(anomaly detection, DMS) is calculated from:

  weekly_growth, 2_week_growth, 4_week_growth
  rolling_mean_4/8/12, rolling_std_4/8/12
  z_score
  demand_ratio_to_average
  consecutive_growth_weeks
  seasonal_baseline / seasonal_deviation (same ISO week, prior years)
"""

import numpy as np
import pandas as pd

MIN_VOLUME_THRESHOLD = 15  # villages below this baseline demand get growth dampened


def _pct_growth(current, previous):
    denom = np.maximum(previous.abs(), 1)
    return (current - previous) / denom * 100


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["village_id", "year", "week"]).copy()
    g = df.groupby("village_id")

    df["previous_demand"] = g["demand"].shift(1)
    df["demand_2w_ago"] = g["demand"].shift(2)
    df["demand_4w_ago"] = g["demand"].shift(4)

    df["weekly_growth"] = _pct_growth(df["demand"], df["previous_demand"])
    df["2_week_growth"] = _pct_growth(df["demand"], df["demand_2w_ago"])
    df["4_week_growth"] = _pct_growth(df["demand"], df["demand_4w_ago"])

    # consecutive weeks of positive growth (momentum persistence)
    is_up = (df["weekly_growth"] > 0).astype(int)
    grp = df.groupby("village_id")

    def _consecutive(s):
        out, streak = [], 0
        for v in s:
            streak = streak + 1 if v == 1 else 0
            out.append(streak)
        return out

    df["consecutive_growth_weeks"] = grp["weekly_growth"].transform(
        lambda s: _consecutive((s > 0).astype(int).tolist())
    )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["village_id", "year", "week"]).copy()
    g = df.groupby("village_id")["demand"]

    for window in (4, 8, 12):
        # shift(1) so the current week is never included in its own baseline
        df[f"rolling_mean_{window}"] = g.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=max(3, w // 2)).mean()
        )
        df[f"rolling_std_{window}"] = g.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=max(3, w // 2)).std()
        )

    df["rolling_mean"] = df["rolling_mean_12"].combine_first(df["rolling_mean_8"]).combine_first(df["rolling_mean_4"])
    df["rolling_std"] = df["rolling_std_12"].combine_first(df["rolling_std_8"]).combine_first(df["rolling_std_4"])
    return df


def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_std = df["rolling_std"].replace(0, np.nan)
    df["z_score"] = (df["demand"] - df["rolling_mean"]) / safe_std
    df["z_score"] = df["z_score"].fillna(0)
    return df


def add_intensity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_baseline = df["rolling_mean"].replace(0, np.nan)
    df["demand_ratio_to_average"] = df["demand"] / safe_baseline
    df["demand_ratio_to_average"] = df["demand_ratio_to_average"].fillna(1.0)
    return df


def add_seasonal_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compare against the same ISO week in prior years where available,
    to avoid flagging predictable seasonal upswings as anomalies."""
    df = df.copy()
    lookup = df.set_index(["village_id", "year", "week"])["demand"]

    def prior_year_value(row, years_back):
        key = (row["village_id"], row["year"] - years_back, row["week"])
        return lookup.get(key, np.nan)

    df["same_week_last_year"] = df.apply(lambda r: prior_year_value(r, 1), axis=1)
    df["same_week_2y_ago"] = df.apply(lambda r: prior_year_value(r, 2), axis=1)

    seasonal_vals = df[["same_week_last_year", "same_week_2y_ago"]]
    df["seasonal_baseline"] = seasonal_vals.mean(axis=1)
    df["seasonal_deviation"] = np.where(
        df["seasonal_baseline"].notna() & (df["seasonal_baseline"] > 0),
        (df["demand"] - df["seasonal_baseline"]) / df["seasonal_baseline"] * 100,
        np.nan,
    )
    return df


def apply_low_volume_safeguard(df: pd.DataFrame) -> pd.DataFrame:
    """Dampen growth-based scores for villages whose baseline demand is
    tiny, so that e.g. 1 -> 2 applications doesn't read as a 100% crisis
    spike. Adds a `volume_dampener` in [0, 1] that downstream scoring
    multiplies growth-derived components by."""
    df = df.copy()
    baseline = df["rolling_mean"].fillna(df["demand"])
    df["volume_dampener"] = np.clip(baseline / MIN_VOLUME_THRESHOLD, 0.15, 1.0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_growth_features(df)
    df = add_rolling_features(df)
    df = add_zscore(df)
    df = add_intensity(df)
    df = add_seasonal_baseline(df)
    df = apply_low_volume_safeguard(df)
    return df


if __name__ == "__main__":
    weekly = pd.read_csv("data/processed/village_weekly.csv")
    features = engineer_features(weekly)
    features.to_csv("data/processed/features.csv", index=False)
    print(f"Wrote {len(features)} rows with engineered features to data/processed/features.csv")
    print(features.tail())
