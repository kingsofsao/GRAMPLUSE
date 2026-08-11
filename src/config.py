"""GRAMPULSE program configuration.

The rural employment framework transitioned from MGNREGA to the
Viksit Bharat – Guarantee for Rozgar and Ajeevika Mission (Gramin)
(VB-G RAM G) Act, 2025, which came into force on 1 July 2026.

Historical MGNREGA observations may still be useful for baseline/trend
analysis. Current observations should be sourced from the VB-G RAM G MIS.
"""

from datetime import date

PROGRAM_NAME = "Viksit Bharat – Guarantee for Rozgar and Ajeevika Mission (Gramin)"
PROGRAM_SHORT_NAME = "VB-G RAM G"
ACT_NAME = "VB-G RAM G Act, 2025"
ACT_EFFECTIVE_DATE = date(2026, 7, 1)

HISTORICAL_PROGRAM_NAME = "MGNREGA (historical)"
EMPLOYMENT_DEMAND_LABEL = "Employment demand"

# The Act provides 125 days of guaranteed wage employment per rural
# household in a financial year. This is descriptive UI context, not a
# model feature or an assumption about the demand data.
GUARANTEED_DAYS = 125

DEFAULT_DATA_FILENAME = "vb_gram_g_raw.csv"
LEGACY_DATA_FILENAME = "mgnrega_raw.csv"


def program_for_date(value):
    """Return the statutory framework associated with an observation date."""
    if value is None:
        return PROGRAM_SHORT_NAME
    try:
        observation_date = value.date() if hasattr(value, "date") else value
        return (
            PROGRAM_SHORT_NAME
            if observation_date >= ACT_EFFECTIVE_DATE
            else HISTORICAL_PROGRAM_NAME
        )
    except Exception:
        return PROGRAM_SHORT_NAME
