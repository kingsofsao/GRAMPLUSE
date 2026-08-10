# GRAMPULSE

AI-powered early-warning system that analyzes MGNREGA job-demand data to
flag villages where economic distress may be building, before it shows up
in visible crisis indicators.

Instead of asking *"which village has high demand?"*, GRAMPULSE asks
*"whose demand is rising unusually fast, and how risky is that rise?"* —
producing a **Distress Momentum Score (DMS, 0–100)** per village, a risk
category, a plain-language explanation, and a recommended administrative
action.

> ⚠️ Scientifically honest framing: elevated MGNREGA demand is a *signal*
> worth reviewing, not proof of distress on its own (it can also reflect
> seasonal patterns, drought, migration, or reporting changes). GRAMPULSE
> recommends **administrative review**, not conclusions about cause.

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate synthetic sample data (swap for a real MGNREGA MIS export later)
python data/raw/generate_sample_data.py

# 2. Run the full pipeline from the command line
python src/run_pipeline.py

# 3. Launch the interactive dashboard
streamlit run dashboard/app.py
```

Outputs land in:
- `data/processed/village_weekly.csv` — cleaned weekly time series
- `data/processed/village_risk.csv` — full history with all scores
- `outputs/rankings.csv` — latest-week ranking, all villages
- `outputs/alerts.csv` — latest-week HIGH/EXTREME villages only

## How it works

```
MGNREGA CSV → Clean & standardize → Weekly village time series
  → Feature engineering (growth, rolling stats, z-score, seasonality)
  → Anomaly detection (rolling Z-score + Isolation Forest)
  → Distress Momentum Score (Speed + Intensity + Abnormality)
  → Risk classification (LOW / MODERATE / HIGH / EXTREME)
  → Recommendation engine + explanations
  → Streamlit dashboard (Overview, Risk Map, Village Detail, Ranking, Explainability)
```

### DMS formula

```
DMS = 0.40 × Speed + 0.30 × Intensity + 0.30 × Abnormality
```

- **Speed** — how fast weekly demand is growing (dampened for very
  low-volume villages so tiny absolute changes don't read as huge % spikes)
- **Intensity** — current demand vs. rolling historical baseline
- **Abnormality** — blended rolling Z-score + Isolation Forest signal

Weights, the speed/intensity curves, and risk thresholds all live in
`src/dms.py` and `src/risk.py` as plain config — recalibrate them against
real historical outcomes before treating this as more than a prototype.

## Project structure

```
GRAMPULSE/
├── data/
│   ├── raw/                 # input CSVs + synthetic data generator
│   └── processed/           # cleaned + feature-engineered outputs
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── anomaly_detection.py
│   ├── dms.py
│   ├── risk.py
│   ├── recommendations.py
│   ├── explainability.py
│   └── run_pipeline.py      # orchestrates the full flow
├── dashboard/
│   └── app.py                # Streamlit UI, 5 sections
├── tests/
│   └── test_dms.py
├── outputs/                  # rankings.csv, alerts.csv (generated)
└── requirements.txt
```

## Using your own data

Any CSV with these columns works (case-insensitive):

```
state, district, block, panchayat, village, date, demand
```

Upload it directly in the dashboard sidebar, or pass it to the pipeline:

```bash
python src/run_pipeline.py --input path/to/your_export.csv
```

## What's MVP vs. future work

**Included (MVP):** cleaning, weekly aggregation, rolling stats, Z-score +
Isolation Forest anomaly detection, DMS, risk classification, ranking,
trend charts with anomaly markers, explainability panel, recommendations,
low-volume safeguard, basic seasonal baseline (same ISO week, prior years).

**Not included yet — reasonable next steps:** live MGNREGA MIS
scraping/API integration, an actual geographic map (needs village
lat/lon), SHAP-based explanations, rainfall/drought/satellite features,
backtesting against real historical distress events, SMS/WhatsApp alerts.

## Validating the model

Don't just claim "our AI detected distress" — back it up:
- If you have labeled historical distress events: compute precision,
  recall, F1, false-positive rate.
- If you don't: backtest — train on an earlier period, hide a later
  period, and check whether villages flagged HIGH/EXTREME actually saw
  unusually high demand in the held-out weeks.
