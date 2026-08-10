"""
generate_sample_data.py
------------------------
Generates a synthetic MGNREGA-style village-level weekly job-demand dataset
so GRAMPULSE can be developed, demoed, and tested without needing live
portal access first. Replace this with real MGNREGA MIS exports/scrapes
before the final demo if possible - real data will make the pitch stronger.

Simulates:
  - Normal villages (stable, seasonal demand)
  - Slow-rising villages (gradual distress)
  - Spike villages (sudden abnormal demand increase -> should be flagged EXTREME)
  - Low-volume villages (to test the minimum-volume safeguard)

Output: data/raw/mgnrega_raw.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

STATE = "Tamil Nadu"
DISTRICTS = {
    "Salem": ["Yercaud", "Attur"],
    "Villupuram": ["Gingee", "Tindivanam"],
}

N_WEEKS = 104  # 2 years of weekly history
START_DATE = datetime(2024, 1, 1)

VILLAGE_PROFILES = [
    # (name, block_index_in_district, district, type, base_demand)
    ("Kumarapuram", "Yercaud", "Salem", "normal", 120),
    ("Nallur", "Yercaud", "Salem", "normal", 90),
    ("Servarayan", "Attur", "Salem", "slow_rise", 100),
    ("Periyakottai", "Attur", "Salem", "spike", 150),
    ("Vadakku Agraharam", "Gingee", "Villupuram", "normal", 60),
    ("Melpakkam", "Gingee", "Villupuram", "low_volume", 8),
    ("Thenmalai", "Tindivanam", "Villupuram", "spike", 130),
    ("Alagapuram", "Tindivanam", "Villupuram", "slow_rise", 75),
    ("Chinnakoundampalayam", "Attur", "Salem", "normal", 110),
    ("Rajapalayam Colony", "Gingee", "Villupuram", "normal", 95),
]


def seasonal_multiplier(week_idx):
    """MGNREGA demand tends to rise in the dry/lean agricultural months.
    Approximate a yearly seasonal wave."""
    return 1.0 + 0.25 * np.sin(2 * np.pi * (week_idx % 52) / 52)


def simulate_village(name, block, district, vtype, base):
    rows = []
    demand = base
    for w in range(N_WEEKS):
        date = START_DATE + timedelta(weeks=w)
        season = seasonal_multiplier(w)
        noise = np.random.normal(0, base * 0.06)

        if vtype == "normal":
            demand = base * season + noise

        elif vtype == "slow_rise":
            drift = base * 0.006 * w  # gradual distress build-up
            demand = (base + drift) * season + noise

        elif vtype == "spike":
            demand = base * season + noise
            # Inject a sudden abnormal spike in the last ~6 weeks
            if w >= N_WEEKS - 6:
                spike_factor = 1.5 + 0.35 * (w - (N_WEEKS - 6))
                demand = base * spike_factor + noise

        elif vtype == "low_volume":
            demand = max(0, base + np.random.normal(0, 2))
            if w >= N_WEEKS - 2:
                demand = demand * 2  # small-number pct spike, should be dampened

        demand = max(0, round(demand))
        rows.append({
            "state": STATE,
            "district": district,
            "block": block,
            "panchayat": name + " GP",
            "village": name,
            "date": date.strftime("%Y-%m-%d"),
            "demand": int(demand),
        })
    return rows


def main():
    all_rows = []
    for name, block, district, vtype, base in VILLAGE_PROFILES:
        all_rows.extend(simulate_village(name, block, district, vtype, base))

    df = pd.DataFrame(all_rows)

    # Introduce a few realistic messiness artifacts on purpose, so the
    # preprocessing pipeline has something real to clean:
    # 1) a few missing values
    missing_idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[missing_idx, "demand"] = np.nan

    # 2) a duplicated record
    df = pd.concat([df, df.iloc[[3, 40]]], ignore_index=True)

    # 3) inconsistent village-name casing/spacing for one village
    mask = df["village"] == "Kumarapuram"
    dup_rows = df[mask].sample(5, random_state=1).copy()
    dup_rows["village"] = dup_rows["village"].apply(
        lambda v: "KUMARAPURAM" if np.random.rand() > 0.5 else "Kumara Puram"
    )
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4) inconsistent date formats for a handful of rows
    fmt_idx = np.random.choice(df.index, size=6, replace=False)
    def messy_date(d):
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.strftime("%d/%m/%y")
    df.loc[fmt_idx, "date"] = df.loc[fmt_idx, "date"].apply(messy_date)

    df.to_csv("data/raw/mgnrega_raw.csv", index=False)
    print(f"Wrote {len(df)} rows to data/raw/mgnrega_raw.csv")


if __name__ == "__main__":
    main()
