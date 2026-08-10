"""
recommendations.py
--------------------
Maps risk -> recommended administrative action.

Deliberately phrased as "review"/"assess" language rather than asserting
a confirmed cause (e.g. never "this village is in financial crisis"),
since elevated MGNREGA demand can stem from several different underlying
drivers (seasonal patterns, drought, migration, reporting changes, etc).
DMS flags where to look, not what is definitely happening.
"""

import pandas as pd

RECOMMENDATIONS = {
    "LOW": "Continue routine monitoring. No administrative action required at this time.",
    "MODERATE": "Increase monitoring frequency. Review recent employment demand trends for this village.",
    "HIGH": "Review pending employment demand and wage payment status. Assess availability of MGNREGA "
            "work and consider additional employment allocation.",
    "EXTREME": "Recommend immediate administrative review. Prioritize this village for on-ground "
               "assessment of employment demand and investigate local socioeconomic indicators before "
               "concluding on cause.",
}


def recommend_action(risk: str) -> str:
    return RECOMMENDATIONS.get(risk, RECOMMENDATIONS["MODERATE"])


def add_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recommendation"] = df["risk"].apply(recommend_action)
    return df


if __name__ == "__main__":
    classified = pd.read_csv("data/processed/village_risk.csv")
    result = add_recommendations(classified)
    result.to_csv("data/processed/village_risk.csv", index=False)
    print(result[["village", "risk", "recommendation"]].tail(5).to_string(index=False))
