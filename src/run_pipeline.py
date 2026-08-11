"""
run_pipeline.py
-----------------
Runs the full GRAMPULSE pipeline end to end:

  raw CSV -> clean -> weekly series -> features -> anomalies -> DMS
  -> risk -> recommendations -> explanations -> rankings.csv / alerts.csv

Usage:
    python src/run_pipeline.py [--input data/raw/vb_gram_g_raw.csv]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from data_loader import load_raw_csv
from preprocessing import clean_pipeline
from feature_engineering import engineer_features
from anomaly_detection import detect_anomalies
from dms import calculate_dms
from risk import add_risk_classification
from recommendations import add_recommendations
from explainability import add_explanations


def run(input_path: str, output_dir: str = "outputs", processed_dir: str = "data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"[1/8] Loading raw data from {input_path} ...")
    raw = load_raw_csv(input_path)

    print("[2/8] Cleaning & building weekly village time series ...")
    weekly = clean_pipeline(raw)
    weekly.to_csv(f"{processed_dir}/village_weekly.csv", index=False)

    print("[3/8] Engineering features (growth, rolling stats, z-score, seasonality) ...")
    features = engineer_features(weekly)

    print("[4/8] Running anomaly detection (Z-score + Isolation Forest) ...")
    anomalies = detect_anomalies(features)

    print("[5/8] Calculating Distress Momentum Score ...")
    scored = calculate_dms(anomalies)

    print("[6/8] Classifying risk levels ...")
    classified = add_risk_classification(scored)

    print("[7/8] Generating recommendations & explanations ...")
    result = add_recommendations(classified)
    result = add_explanations(result)

    print("[8/8] Writing outputs ...")
    result.to_csv(f"{processed_dir}/village_risk.csv", index=False)

    # Latest week per village = current state, this is what the dashboard/rankings use
    latest = (
        result.sort_values(["village_id", "year", "week"])
        .groupby("village_id", as_index=False)
        .tail(1)
        .sort_values("dms", ascending=False)
    )
    latest.to_csv(f"{output_dir}/rankings.csv", index=False)

    alerts = latest[latest["risk"].isin(["HIGH", "EXTREME"])]
    alerts.to_csv(f"{output_dir}/alerts.csv", index=False)

    print(f"\nDone. {len(result)} village-weeks processed.")
    print(f"  Full history:      {processed_dir}/village_risk.csv")
    print(f"  Latest rankings:   {output_dir}/rankings.csv")
    print(f"  Active alerts:     {output_dir}/alerts.csv  ({len(alerts)} villages HIGH/EXTREME)")
    print("\nRisk distribution (latest week per village):")
    print(latest["risk"].value_counts().to_string())

    return result, latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/vb_gram_g_raw.csv")
    args = parser.parse_args()
    run(args.input)
