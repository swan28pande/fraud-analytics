"""Shared paths and helpers used by the analysis scripts."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "transactions.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = ROOT / "model.pkl"

NUMERIC_FEATURES = [
    "amount", "hour_of_day", "day_of_week", "distance_from_home",
    "previous_transaction_gap_hours", "avg_amount_last_30d", "amount_to_avg_ratio",
]
CATEGORICAL_FEATURES = ["merchant_category"]
BINARY_FEATURES = ["is_foreign"]
TARGET = "is_fraud"

# The dataset has no timestamp, so transactions are treated as arriving in
# file order and split evenly into this many daily batches.
N_DAYS = 30


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python data/generate_data.py` first.")
    df = pd.read_csv(path)
    df["amount_to_avg_ratio"] = df["amount"] / df["avg_amount_last_30d"].clip(lower=1)
    return df


def assign_days(df: pd.DataFrame, n_days: int = N_DAYS) -> pd.Series:
    """Map each row to a simulated day number (1..n_days) based on its position."""
    return (pd.Series(range(len(df)), index=df.index) * n_days // len(df)) + 1
