"""
explainability.py
-------------------
Turns the numeric scores into a human-readable explanation, so the
system never just outputs "DMS = 91" and stops. This is what the poster
calls "AI explanations for each flagged village" (XAI).
"""

import pandas as pd


def _fmt_pct(x):
    if pd.isna(x):
        return "unavailable"
    return f"{x:.0f}%"


def explain_row(row: pd.Series) -> str:
    reasons = []

    if pd.notna(row.get("weekly_growth")) and row["weekly_growth"] > 5:
        reasons.append(f"Demand increased {_fmt_pct(row['weekly_growth'])} in the latest week")

    if pd.notna(row.get("demand_ratio_to_average")) and row["demand_ratio_to_average"] > 1.1:
        pct_above = (row["demand_ratio_to_average"] - 1) * 100
        reasons.append(f"Current demand is {pct_above:.0f}% above its historical baseline")

    if pd.notna(row.get("z_score")) and row["z_score"] > 1.5:
        reasons.append(f"Current demand is {row['z_score']:.1f} standard deviations above baseline")

    if row.get("anomaly_detected"):
        reasons.append("Statistical/ML anomaly detection flagged this week as abnormal")

    if pd.notna(row.get("consecutive_growth_weeks")) and row["consecutive_growth_weeks"] >= 3:
        reasons.append(f"Momentum has increased for {int(row['consecutive_growth_weeks'])} consecutive weeks")

    if not reasons:
        reasons.append("No significant deviation from normal demand patterns detected")

    return " · ".join(reasons)


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["explanation"] = df.apply(explain_row, axis=1)
    return df


if __name__ == "__main__":
    result = pd.read_csv("data/processed/village_risk.csv")
    explained = add_explanations(result)
    explained.to_csv("data/processed/village_risk.csv", index=False)
    print(explained[["village", "risk", "explanation"]].tail(5).to_string(index=False))
