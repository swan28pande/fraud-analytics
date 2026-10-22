"""Generate a synthetic credit card transaction dataset with a ~2% fraud rate.

Usage:
    python data/generate_data.py [--rows 100000] [--out data/transactions.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 100_000
FRAUD_RATE = 0.02
DEFAULT_OUT = Path(__file__).resolve().parent / "transactions.csv"

# Category mix for legitimate vs. fraudulent transactions.
CATEGORIES = [
    "grocery", "gas", "restaurant", "retail", "online",
    "travel", "entertainment", "electronics", "health", "utilities",
]
LEGIT_WEIGHTS = np.array([0.20, 0.12, 0.15, 0.14, 0.13, 0.04, 0.07, 0.04, 0.06, 0.05])
FRAUD_WEIGHTS = np.array([0.04, 0.06, 0.04, 0.10, 0.30, 0.14, 0.06, 0.20, 0.02, 0.04])

# Typical spend per category (log-normal median) for legit transactions.
CATEGORY_MEDIAN = {
    "grocery": 45, "gas": 40, "restaurant": 35, "retail": 60, "online": 55,
    "travel": 220, "entertainment": 50, "electronics": 180, "health": 70, "utilities": 110,
}


def _hours(rng, n, fraud):
    """Legit traffic peaks during the day; fraud skews to late night."""
    if fraud:
        probs = np.array([6, 7, 7, 7, 6, 4, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6], float)
    else:
        probs = np.array([1, 0.6, 0.4, 0.3, 0.3, 0.6, 1.5, 3, 4.5, 5, 5.5, 6, 6.5, 6, 5.5, 5.5, 6, 6.5, 6.5, 6, 5, 4, 3, 2], float)
    return rng.choice(24, size=n, p=probs / probs.sum())


def _block(rng, n, fraud):
    weights = FRAUD_WEIGHTS if fraud else LEGIT_WEIGHTS
    category = rng.choice(CATEGORIES, size=n, p=weights / weights.sum())
    medians = np.array([CATEGORY_MEDIAN[c] for c in category])

    # Customer's recent typical spend, loosely tied to the category.
    avg_30d = np.round(medians * rng.lognormal(0.0, 0.35, n), 2)

    if fraud:
        # Fraud amounts are usually well above the customer's norm, with some small "card testing" charges.
        multiplier = rng.lognormal(1.2, 0.8, n)
        card_test = rng.random(n) < 0.15
        amount = np.where(card_test, rng.uniform(0.5, 5, n), avg_30d * multiplier)
        distance = rng.lognormal(3.6, 1.3, n)
        is_foreign = (rng.random(n) < 0.35).astype(int)
        gap = rng.exponential(2.5, n)
        day = rng.choice(7, size=n, p=[0.13, 0.13, 0.13, 0.13, 0.16, 0.16, 0.16])
    else:
        amount = avg_30d * rng.lognormal(0.0, 0.55, n)
        distance = rng.lognormal(1.8, 1.0, n)
        is_foreign = (rng.random(n) < 0.04).astype(int)
        gap = rng.exponential(22, n)
        day = rng.choice(7, size=n, p=[0.14, 0.14, 0.14, 0.14, 0.15, 0.15, 0.14])

    return pd.DataFrame({
        "amount": np.round(np.clip(amount, 0.5, 25_000), 2),
        "merchant_category": category,
        "hour_of_day": _hours(rng, n, fraud),
        "day_of_week": day,
        "distance_from_home": np.round(np.clip(distance, 0, 12_000), 2),
        "is_foreign": is_foreign,
        "previous_transaction_gap_hours": np.round(np.clip(gap, 0.01, 720), 2),
        "avg_amount_last_30d": avg_30d,
        "is_fraud": int(fraud),
    })


def generate(n_rows: int = N_ROWS, fraud_rate: float = FRAUD_RATE, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n_rows * fraud_rate))
    df = pd.concat([_block(rng, n_rows - n_fraud, False), _block(rng, n_fraud, True)], ignore_index=True)

    # Shuffle so fraud is spread throughout the file (transaction order is used as time downstream).
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(1, n_rows + 1)])
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = generate(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%} ({df['is_fraud'].sum():,} fraudulent)")


if __name__ == "__main__":
    main()
