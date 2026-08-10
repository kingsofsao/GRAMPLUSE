"""
risk.py
--------
Maps a DMS value to a risk category. Thresholds are configuration so
they can be tuned per-state/district or recalibrated later.
"""

import pandas as pd

RISK_THRESHOLDS = [
    (0, 40, "LOW", "🟢"),
    (40, 60, "MODERATE", "🟡"),
    (60, 80, "HIGH", "🟠"),
    (80, 101, "EXTREME", "🔴"),
]


def classify_risk(dms: float) -> str:
    for low, high, label, _ in RISK_THRESHOLDS:
        if low <= dms < high:
            return label
    return "EXTREME"


def risk_emoji(label: str) -> str:
    for _, _, l, emoji in RISK_THRESHOLDS:
        if l == label:
            return emoji
    return ""


def add_risk_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk"] = df["dms"].apply(classify_risk)
    df["risk_emoji"] = df["risk"].apply(risk_emoji)
    return df


if __name__ == "__main__":
    scored = pd.read_csv("data/processed/village_risk.csv")
    classified = add_risk_classification(scored)
    classified.to_csv("data/processed/village_risk.csv", index=False)
    print(classified["risk"].value_counts())
