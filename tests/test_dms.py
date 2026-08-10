"""
Basic sanity tests for the DMS engine. Run with:  pytest tests/test_dms.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from dms import growth_to_speed_score, ratio_to_intensity_score, calculate_dms
from risk import classify_risk
from anomaly_detection import zscore_to_score_0_100


def test_speed_score_zero_growth():
    assert growth_to_speed_score(0) == 0

def test_speed_score_high_growth_caps_at_100():
    assert growth_to_speed_score(80) == 100

def test_speed_score_matches_example_from_spec():
    # spec example: 50% growth -> 100 speed score
    assert growth_to_speed_score(50) == 100

def test_intensity_score_at_baseline():
    assert ratio_to_intensity_score(1.0) == 10

def test_intensity_score_high_ratio():
    # spec example: ratio 1.8 -> high intensity
    score = ratio_to_intensity_score(1.8)
    assert 75 <= score <= 85

def test_zscore_mapping_matches_spec_example():
    # spec example: z=4.0 -> should map to a high abnormality score (80/100)
    s = zscore_to_score_0_100(pd.Series([4.0])).iloc[0]
    assert abs(s - 80) < 0.01

def test_risk_classification_bands():
    assert classify_risk(20) == "LOW"
    assert classify_risk(50) == "MODERATE"
    assert classify_risk(70) == "HIGH"
    assert classify_risk(91) == "EXTREME"

def test_calculate_dms_matches_worked_example():
    # From the spec's worked example: speed=90, intensity=85, abnormality=95 -> DMS=90
    df = pd.DataFrame([{
        "weekly_growth": 50, "volume_dampener": 1.0,
        "demand_ratio_to_average": 1.8,
        "abnormality_score": 95,
    }])
    # override speed/intensity directly is not how calculate_dms works (it derives
    # them from growth/ratio), so instead check the derived DMS is in the right
    # risk band rather than an exact match to the poster's illustrative numbers.
    scored = calculate_dms(df)
    assert scored["dms"].iloc[0] >= 80  # should land in EXTREME
    assert classify_risk(scored["dms"].iloc[0]) == "EXTREME"

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
