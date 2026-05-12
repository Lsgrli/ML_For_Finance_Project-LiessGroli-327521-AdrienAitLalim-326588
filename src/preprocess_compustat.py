from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

INPUT_FILE = RAW_DATA_DIR / "CompFirmCharac.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "compustat.parquet"


# ---------------------------------------------------------------------
# Cleaning choices
# ---------------------------------------------------------------------

MAX_MISSING_SHARE = 0.80


# ---------------------------------------------------------------------
# Load Compustat firm characteristics
# ---------------------------------------------------------------------

compustat = pd.read_csv(INPUT_FILE, low_memory=False)


# ---------------------------------------------------------------------
# Clean column names
# ---------------------------------------------------------------------
# Compustat contains firm-level accounting variables.
# The two most important columns for now are:
# - cusip: firm/security identifier, useful for merging with CRSP
# - datadate: accounting/reporting date

compustat.columns = [
    col.strip().lower().replace(" ", "").replace("_", "")
    for col in compustat.columns
]


# ---------------------------------------------------------------------
# Rename important columns
# ---------------------------------------------------------------------

rename_map = {}

for col in compustat.columns:
    normalized = col.lower().replace("_", "").replace(" ", "")

    if normalized == "cusip":
        rename_map[col] = "cusip"

    elif normalized in ["datadate", "date", "reportdate"]:
        rename_map[col] = "date"

    elif normalized in ["tic", "ticker"]:
        rename_map[col] = "ticker"

    elif normalized in ["conm", "companyname"]:
        rename_map[col] = "company_name"

    elif normalized in ["gvkey"]:
        rename_map[col] = "gvkey"

compustat = compustat.rename(columns=rename_map)


# ---------------------------------------------------------------------
# Check required columns
# ---------------------------------------------------------------------

required_columns = ["cusip", "date"]

missing_columns = [
    col for col in required_columns
    if col not in compustat.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}. "
        f"Available columns are: {compustat.columns.tolist()}"
    )


# ---------------------------------------------------------------------
# Convert important columns to proper types
# ---------------------------------------------------------------------

compustat["date"] = pd.to_datetime(
    compustat["date"],
    errors="coerce",
    dayfirst=True
)

compustat["cusip"] = compustat["cusip"].astype(str).str.strip()

if "ticker" in compustat.columns:
    compustat["ticker"] = compustat["ticker"].astype(str).str.strip()

if "company_name" in compustat.columns:
    compustat["company_name"] = compustat["company_name"].astype(str).str.strip()

if "gvkey" in compustat.columns:
    compustat["gvkey"] = compustat["gvkey"].astype(str).str.strip()


# ---------------------------------------------------------------------
# Remove rows that cannot be used
# ---------------------------------------------------------------------
# We need a firm identifier and a reporting date.
# Without these, we cannot merge Compustat with stock-month CRSP data.

compustat = compustat.dropna(subset=["cusip", "date"])

compustat = compustat[
    compustat["cusip"].notna()
    & (compustat["cusip"] != "")
    & (compustat["cusip"].str.lower() != "nan")
]


# ---------------------------------------------------------------------
# Convert accounting variables to numeric when possible
# ---------------------------------------------------------------------
# Compustat has many accounting columns. Some are numeric, some are text.
# We keep identifiers as text and try to convert the rest into numbers.

identifier_columns = [
    "cusip",
    "date",
    "ticker",
    "company_name",
    "gvkey",
]

identifier_columns = [
    col for col in identifier_columns
    if col in compustat.columns
]

candidate_feature_columns = [
    col for col in compustat.columns
    if col not in identifier_columns
]

for col in candidate_feature_columns:
    compustat[col] = pd.to_numeric(compustat[col], errors="coerce")


# ---------------------------------------------------------------------
# Remove feature columns with too many missing values
# ---------------------------------------------------------------------
# If a column is missing for almost all firms, it is probably not useful.
# Here we keep columns with at most 80% missing values.

missing_share_by_column = compustat[candidate_feature_columns].isna().mean()

kept_feature_columns = missing_share_by_column[
    missing_share_by_column <= MAX_MISSING_SHARE
].index.tolist()


# ---------------------------------------------------------------------
# Remove duplicate firm-date observations
# ---------------------------------------------------------------------
# We want one row per firm and reporting date.

compustat = compustat.sort_values(["cusip", "date"])

compustat = compustat.drop_duplicates(
    subset=["cusip", "date"],
    keep="last"
)


# ---------------------------------------------------------------------
# Keep useful columns
# ---------------------------------------------------------------------

columns_to_keep = identifier_columns + kept_feature_columns

compustat = compustat[columns_to_keep]


# ---------------------------------------------------------------------
# Save cleaned Compustat data
# ---------------------------------------------------------------------

compustat = compustat.sort_values(["cusip", "date"]).reset_index(drop=True)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

compustat.to_parquet(OUTPUT_FILE, index=False)

print("Saved cleaned Compustat data.")
print(f"Output file: {OUTPUT_FILE}")
print(f"Number of firm-date observations: {len(compustat)}")
print(f"Number of firms by CUSIP: {compustat['cusip'].nunique()}")
print(f"Date range: {compustat['date'].min().date()} to {compustat['date'].max().date()}")
print(f"Number of accounting feature columns kept: {len(kept_feature_columns)}")