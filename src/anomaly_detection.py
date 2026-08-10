"""
anomaly_detection.py
----------------------
Two anomaly-detection methods, as recommended in the analysis:

  Method A - Rolling Z-score        -> explainable, simple, good default
  Method B - Isolation Forest       -> catches nonlinear / multi-feature
                                        combinations the z-score alone misses

Both scores are normalized to 0-100 and combined into a single
`abnormality_score` used by the DMS engine.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ISO_FOREST_FEATURES = [
    "weekly_growth",
    "4_week_growth",
    "z_score",
    "demand_ratio_to_average",
    "consecutive_growth_weeks",
]


def zscore_to_score_0_100(z: pd.Series) -> pd.Series:
    """Map a z-score to 0-100. z<=0 -> 0, z>=5 -> 100, linear between."""
    return np.clip(z, 0, 5) / 5 * 100


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.08, random_state: int = 42) -> pd.Series:
    """Returns a 0-100 abnormality score per row from Isolation Forest.
    Rows with insufficient history (NaNs in the feature set) get a
    neutral score of 0 rather than being dropped, since the DMS engine
    needs one row per village-week."""
    feat_df = df[ISO_FOREST_FEATURES].copy()
    valid_mask = feat_df.notna().all(axis=1)

    scores = pd.Series(0.0, index=df.index)
    if valid_mask.sum() < 10:
        # not enough data yet to train a meaningful model
        return scores

    X = feat_df.loc[valid_mask].fillna(0)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    model.fit(X)

    # decision_function: higher = more normal, lower = more anomalous.
    raw = model.decision_function(X)
    # invert and min-max scale to 0-100 within this batch
    inverted = -raw
    rng = inverted.max() - inverted.min()
    if rng == 0:
        normalized = np.zeros_like(inverted)
    else:
        normalized = (inverted - inverted.min()) / rng * 100

    scores.loc[valid_mask] = normalized
    return scores


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zscore_abnormality"] = zscore_to_score_0_100(df["z_score"])
    df["isoforest_abnormality"] = run_isolation_forest(df)

    # Combine: average of the two, but let a strong single-signal (z-score
    # explainability) still dominate when Isolation Forest hasn't got
    # enough history yet.
    df["abnormality_score"] = (
        0.6 * df["zscore_abnormality"] + 0.4 * df["isoforest_abnormality"]
    ).clip(0, 100)

    df["anomaly_detected"] = df["abnormality_score"] >= 60
    return df


if __name__ == "__main__":
    features = pd.read_csv("data/processed/features.csv")
    result = detect_anomalies(features)
    result.to_csv("data/processed/anomalies.csv", index=False)
    flagged = result[result["anomaly_detected"]]
    print(f"Flagged {len(flagged)} of {len(result)} village-weeks as anomalous.")
