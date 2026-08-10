"""
dms.py
-------
The Distress Momentum Score (DMS) engine.

DMS = w1*Speed + w2*Intensity + w3*Abnormality   (each sub-score in 0-100)

Speed       -> how fast demand is rising (weekly_growth, dampened for
               low-volume villages)
Intensity   -> how high current demand is vs. its historical baseline
Abnormality -> combined z-score + Isolation Forest signal from
               anomaly_detection.py

Weights and thresholds are configuration, not hardcoded constants, so
they can be recalibrated against historical outcomes or domain-expert
review without touching the scoring logic.
"""

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"speed": 0.40, "intensity": 0.30, "abnormality": 0.30}

# Piecewise-linear growth% -> 0-100 speed score anchor points
SPEED_CURVE = [
    (-100, 0),
    (0, 0),
    (5, 20),
    (10, 35),
    (20, 55),
    (30, 70),
    (50, 100),
    (1000, 100),
]

# Piecewise-linear intensity ratio -> 0-100 intensity score anchor points
INTENSITY_CURVE = [
    (0.0, 0),
    (1.0, 10),
    (1.2, 35),
    (1.5, 60),
    (1.8, 80),
    (2.5, 95),
    (5.0, 100),
]


def _piecewise_interp(x, curve):
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    return float(np.interp(x, xs, ys))


def growth_to_speed_score(growth_pct: float) -> float:
    if pd.isna(growth_pct):
        return 0.0
    return _piecewise_interp(growth_pct, SPEED_CURVE)


def ratio_to_intensity_score(ratio: float) -> float:
    if pd.isna(ratio):
        return 0.0
    return _piecewise_interp(ratio, INTENSITY_CURVE)


def calculate_speed_score(df: pd.DataFrame) -> pd.Series:
    raw = df["weekly_growth"].apply(growth_to_speed_score)
    # apply the low-volume safeguard so tiny villages don't get an
    # extreme score purely from noisy percentage swings
    dampened = raw * df["volume_dampener"]
    return dampened.clip(0, 100)


def calculate_intensity_score(df: pd.DataFrame) -> pd.Series:
    return df["demand_ratio_to_average"].apply(ratio_to_intensity_score).clip(0, 100)


def calculate_dms(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    df = df.copy()

    df["speed_score"] = calculate_speed_score(df)
    df["intensity_score"] = calculate_intensity_score(df)
    # abnormality_score must already exist (from anomaly_detection.detect_anomalies)
    if "abnormality_score" not in df.columns:
        raise ValueError("Run anomaly_detection.detect_anomalies() before calculate_dms().")

    df["dms"] = (
        weights["speed"] * df["speed_score"]
        + weights["intensity"] * df["intensity_score"]
        + weights["abnormality"] * df["abnormality_score"]
    ).round(1)

    df["dms"] = df["dms"].clip(0, 100)
    return df


if __name__ == "__main__":
    anomalies = pd.read_csv("data/processed/anomalies.csv")
    scored = calculate_dms(anomalies)
    scored.to_csv("data/processed/village_risk.csv", index=False)
    print(scored[["village", "date", "speed_score", "intensity_score", "abnormality_score", "dms"]].tail(10))
