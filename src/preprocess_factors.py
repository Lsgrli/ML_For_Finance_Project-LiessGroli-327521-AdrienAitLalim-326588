from pathlib import Path

import pandas as pd
import numpy as np


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

INPUT_FILE = RAW_DATA_DIR / "[usa]_[all_factors]_[monthly]_[vw_cap].csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "factor_regime_features.parquet"


# ---------------------------------------------------------------------
# Rolling windows used to describe the recent factor environment
# ---------------------------------------------------------------------

SHORT_WINDOW = 3
LONG_WINDOW = 12


# ---------------------------------------------------------------------
# Load JKP factor returns
# ---------------------------------------------------------------------

returns = pd.read_csv(INPUT_FILE, low_memory=False)


# ---------------------------------------------------------------------
# Clean column names
# ---------------------------------------------------------------------
# The JKP file should contain factor returns in long format:
# one row = one factor in one month.
#
# Important columns are usually:
# date      = month
# name      = factor name
# ret       = factor return
# n stocks  = number of stocks in the factor portfolio

returns.columns = [col.strip() for col in returns.columns]

rename_map = {}

for col in returns.columns:
    normalized = col.lower().replace("_", "").replace(" ", "")

    if normalized in ["date", "month", "mthcaldt"]:
        rename_map[col] = "date"

    elif normalized in ["name", "factor", "factorname"]:
        rename_map[col] = "factor_name"

    elif normalized in ["ret", "return", "factorreturn"]:
        rename_map[col] = "return"

    elif normalized in ["nstocks", "numberofstocks"]:
        rename_map[col] = "n_stocks"

returns = returns.rename(columns=rename_map)


# ---------------------------------------------------------------------
# Check required columns
# ---------------------------------------------------------------------

required_columns = ["date", "factor_name", "return"]

missing_columns = [
    col for col in required_columns
    if col not in returns.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}. "
        f"Available columns are: {returns.columns.tolist()}"
    )


# ---------------------------------------------------------------------
# Convert important columns to proper types
# ---------------------------------------------------------------------

returns["date"] = pd.to_datetime(
    returns["date"],
    errors="coerce"
)

returns["return"] = pd.to_numeric(
    returns["return"],
    errors="coerce"
)

if "n_stocks" in returns.columns:
    returns["n_stocks"] = pd.to_numeric(
        returns["n_stocks"],
        errors="coerce"
    )


# ---------------------------------------------------------------------
# Remove rows that cannot be used
# ---------------------------------------------------------------------
# We need a date, a factor name, and a factor return.
# Without these three elements, the observation is unusable.

returns = returns.dropna(
    subset=["date", "factor_name", "return"]
)


# ---------------------------------------------------------------------
# Remove duplicate factor-month observations
# ---------------------------------------------------------------------
# We want one row per factor per month.

returns = returns.sort_values(
    ["factor_name", "date"]
)

returns = returns.drop_duplicates(
    subset=["factor_name", "date"],
    keep="last"
)


# ---------------------------------------------------------------------
# Create a factor return panel
# ---------------------------------------------------------------------
# After pivoting:
# one row = one month
# one column = one factor
# value = factor return in that month

return_panel = returns.pivot(
    index="date",
    columns="factor_name",
    values="return"
)

return_panel = return_panel.sort_index()


# ---------------------------------------------------------------------
# Create market-wide anomaly regime features
# ---------------------------------------------------------------------
# These features summarize whether factor/anomaly strategies have
# recently performed well or poorly.
#
# They are common to all stocks in a given month and will later be merged
# with the CRSP stock-month panel using the date.

factor_regime_features = pd.DataFrame(index=return_panel.index)

factor_regime_features["factor_mean_return_3m"] = (
    return_panel
    .rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW)
    .mean()
    .mean(axis=1)
)

factor_regime_features["factor_mean_return_12m"] = (
    return_panel
    .rolling(LONG_WINDOW, min_periods=LONG_WINDOW)
    .mean()
    .mean(axis=1)
)

factor_regime_features["factor_volatility_12m"] = (
    return_panel
    .rolling(LONG_WINDOW, min_periods=LONG_WINDOW)
    .std()
    .mean(axis=1)
)

factor_rolling_mean_12m = return_panel.rolling(
    LONG_WINDOW,
    min_periods=LONG_WINDOW
).mean()

factor_rolling_volatility_12m = return_panel.rolling(
    LONG_WINDOW,
    min_periods=LONG_WINDOW
).std()

factor_rolling_sharpe_12m = (
    factor_rolling_mean_12m / factor_rolling_volatility_12m
)

factor_rolling_sharpe_12m = factor_rolling_sharpe_12m.replace(
    [np.inf, -np.inf],
    np.nan
)

factor_regime_features["factor_sharpe_12m"] = (
    factor_rolling_sharpe_12m.mean(axis=1)
)

factor_regime_features["return_dispersion"] = (
    return_panel.std(axis=1)
)

factor_regime_features["return_dispersion_12m"] = (
    factor_regime_features["return_dispersion"]
    .rolling(LONG_WINDOW, min_periods=LONG_WINDOW)
    .mean()
)

factor_regime_features["number_positive_factors_3m"] = (
    return_panel
    .rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW)
    .mean()
    .gt(0)
    .sum(axis=1)
)

factor_regime_features["number_positive_factors_12m"] = (
    return_panel
    .rolling(LONG_WINDOW, min_periods=LONG_WINDOW)
    .mean()
    .gt(0)
    .sum(axis=1)
)


# ---------------------------------------------------------------------
# Clean final factor regime dataset
# ---------------------------------------------------------------------
# We remove the first months where rolling features are not available.

factor_regime_features = factor_regime_features.reset_index()

factor_regime_features = factor_regime_features.dropna(
    subset=[
        "factor_mean_return_3m",
        "factor_mean_return_12m",
        "factor_volatility_12m",
        "factor_sharpe_12m",
        "return_dispersion_12m",
    ]
)

factor_regime_features = factor_regime_features.sort_values(
    "date"
).reset_index(drop=True)


# ---------------------------------------------------------------------
# Save factor regime features
# ---------------------------------------------------------------------

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

factor_regime_features.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("Saved factor regime features.")
print(f"Output file: {OUTPUT_FILE}")
print(f"Number of monthly observations: {len(factor_regime_features)}")
print(
    "Date range: "
    f"{factor_regime_features['date'].min().date()} "
    f"to {factor_regime_features['date'].max().date()}"
)
print(f"Number of regime features: {factor_regime_features.shape[1] - 1}")