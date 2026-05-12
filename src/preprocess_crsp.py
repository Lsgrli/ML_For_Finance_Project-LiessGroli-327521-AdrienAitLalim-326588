from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

INPUT_FILE = RAW_DATA_DIR / "monthly_crsp.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "crsp_clean.parquet"


# ---------------------------------------------------------------------
# Optional cleaning choices
# ---------------------------------------------------------------------
# By default, we keep the full available CRSP sample.
# If we later want a restricted period, we can set START_DATE manually.
MIN_OBS_PER_STOCK = 24

crsp = pd.read_csv(INPUT_FILE, low_memory=False)

# ---------------------------------------------------------------------
# Clean column names
# ---------------------------------------------------------------------

crsp.columns = [col.strip() for col in crsp.columns]

rename_map = {}

for col in crsp.columns:
    normalized = col.lower().replace("_", "").replace(" ", "")

    if normalized == "permno":
        rename_map[col] = "stock_id"

    elif normalized in ["mthcaldt", "date", "month", "caldt"]:
        rename_map[col] = "date"

    elif normalized in ["mthret", "ret", "return", "monthlyreturn"]:
        rename_map[col] = "stock_return"

    elif normalized == "sprtrn":
        rename_map[col] = "mkt_return"

    elif normalized in ["ticker", "tickersymbol"]:
        rename_map[col] = "ticker"

    elif normalized == "tradingsymbol":
        rename_map[col] = "TradingSymbol"

    elif normalized == "cusip":
        rename_map[col] = "CUSIP"

    elif normalized == "hdrcusip":
        rename_map[col] = "hdr_CUSIP"

    elif normalized == "permco":
        rename_map[col] = "firm_id"

    elif normalized == "siccd":
        rename_map[col] = "SICCD"

    elif normalized == "naics":
        rename_map[col] = "NAICS"

    elif normalized in ["mthprc", "prc", "price"]:
        rename_map[col] = "price"

crsp = crsp.rename(columns=rename_map)

# ---------------------------------------------------------------------
# Convert important columns to proper types
# ---------------------------------------------------------------------

crsp["stock_id"] = pd.to_numeric(crsp["stock_id"], errors="coerce")
crsp["date"] = pd.to_datetime(crsp["date"], errors="coerce")
crsp["stock_return"] = pd.to_numeric(crsp["stock_return"], errors="coerce")

if "mkt_return" in crsp.columns:
    crsp["mkt_return"] = pd.to_numeric(crsp["mkt_return"], errors="coerce")

if "price" in crsp.columns:
    crsp["price"] = pd.to_numeric(crsp["price"], errors="coerce").abs()

if "siccd" in crsp.columns:
    crsp["siccd"] = pd.to_numeric(crsp["siccd"], errors="coerce")

if "naics" in crsp.columns:
    crsp["naics"] = pd.to_numeric(crsp["naics"], errors="coerce")


# ---------------------------------------------------------------------
# Remove rows that cannot be used
# ---------------------------------------------------------------------
# We need a stock identifier, a month, and a return.

crsp = crsp.dropna(subset=["stock_id", "date", "stock_return"])
crsp["stock_id"] = crsp["stock_id"].astype(int)


# ---------------------------------------------------------------------
# Remove impossible returns
# ---------------------------------------------------------------------
# A return lower than -100% is not economically meaningful.

crsp = crsp[crsp["stock_return"] > -1.0]


# ---------------------------------------------------------------------
# Remove duplicate stock-month observations
# ---------------------------------------------------------------------
# We want one row per stock per month.

crsp = crsp.sort_values(["stock_id", "date"])

crsp = crsp.drop_duplicates(
    subset=["stock_id", "date"],
    keep="last"
)


# ---------------------------------------------------------------------
# Keep only stocks with enough return history
# ---------------------------------------------------------------------

obs_per_stock = crsp.groupby("stock_id")["stock_return"].transform("count")

crsp = crsp[obs_per_stock >= MIN_OBS_PER_STOCK]

# ---------------------------------------------------------------------
# Save cleaned data
# ---------------------------------------------------------------------

crsp = crsp.sort_values(["stock_id", "date"]).reset_index(drop=True)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

crsp.to_parquet(OUTPUT_FILE, index=False)

print("Saved cleaned CRSP data.")
print(f"Output file: {OUTPUT_FILE}")
print(f"Number of stock-month observations: {len(crsp)}")
print(f"Number of stocks: {crsp['stock_id'].nunique()}")
print(f"Date range: {crsp['date'].min().date()} to {crsp['date'].max().date()}")