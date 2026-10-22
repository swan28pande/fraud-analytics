"""Simulated daily batch monitoring with KPI reporting.

The dataset has no timestamps, so transactions are treated as arriving in file
order and split into 30 equal daily batches. Each batch is scored with the
trained gradient boosting model (model.pkl) and compared with a trailing
7-day baseline.

Note: the model was trained on a random 75% of this same dataset, so the model
metrics here are optimistic. In production the batches would be unseen data.

Usage:
    python src/monitoring.py            # report for the most recent day
    python src/monitoring.py --day 12   # report for a specific day
"""

import argparse
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import MODEL_PATH, N_DAYS, OUTPUT_DIR, assign_days, load_data

BASELINE_DAYS = 7
THRESHOLD = 0.5
VELOCITY_GAP_HOURS = 1.0
HIGH_RATIO = 3.0
NIGHT_HOURS = range(0, 6)
SIM_START = date(2026, 1, 1)

# Alert thresholds (relative change vs. baseline, or absolute drop for precision).
FRAUD_RATE_ALERT = 0.25
PRECISION_DROP_ALERT = 0.05
RECALL_DROP_ALERT = 0.05


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python src/pipeline.py` first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def model_metrics(batch: pd.DataFrame) -> dict:
    y, p = batch["is_fraud"], batch["predicted"]
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "fpr": safe_div(fp, fp + tn),
    }


def compute_kpis(batch: pd.DataFrame) -> dict:
    fraud = batch[batch["is_fraud"] == 1]
    legit = batch[batch["is_fraud"] == 0]
    foreign = batch[batch["is_foreign"] == 1]
    domestic = batch[batch["is_foreign"] == 0]
    rapid = batch[batch["previous_transaction_gap_hours"] < VELOCITY_GAP_HOURS]
    night = batch[batch["hour_of_day"].isin(NIGHT_HOURS)]
    high_ratio = batch[batch["amount_to_avg_ratio"] > HIGH_RATIO]

    k = {
        "volume": len(batch),
        "total_amount": batch["amount"].sum(),
        "fraud_count": len(fraud),
        "fraud_rate": safe_div(len(fraud), len(batch)),
        "fraud_amount": fraud["amount"].sum(),
        "fraud_amount_share": safe_div(fraud["amount"].sum(), batch["amount"].sum()),
        "avg_fraud_amount": fraud["amount"].mean() if len(fraud) else 0.0,
        "avg_legit_amount": legit["amount"].mean(),
        "p50_amount": batch["amount"].quantile(0.50),
        "p90_amount": batch["amount"].quantile(0.90),
        "p99_amount": batch["amount"].quantile(0.99),
        "p90_fraud_amount": fraud["amount"].quantile(0.90) if len(fraud) else 0.0,
        "foreign_fraud_rate": safe_div(foreign["is_fraud"].sum(), len(foreign)),
        "domestic_fraud_rate": safe_div(domestic["is_fraud"].sum(), len(domestic)),
        "foreign_share": safe_div(len(foreign), len(batch)),
        "velocity_share": safe_div(len(rapid), len(batch)),
        "velocity_fraud_rate": safe_div(rapid["is_fraud"].sum(), len(rapid)),
        "median_gap_fraud": fraud["previous_transaction_gap_hours"].median() if len(fraud) else 0.0,
        "median_gap_legit": legit["previous_transaction_gap_hours"].median(),
        "night_fraud_rate": safe_div(night["is_fraud"].sum(), len(night)),
        "night_fraud_share": safe_div(fraud["hour_of_day"].isin(NIGHT_HOURS).sum(), len(fraud)),
        "median_distance_fraud": fraud["distance_from_home"].median() if len(fraud) else 0.0,
        "median_distance_legit": legit["distance_from_home"].median(),
        "high_ratio_fraud_rate": safe_div(high_ratio["is_fraud"].sum(), len(high_ratio)),
        "avg_fraud_score": batch["score"].mean(),
    }
    k.update(model_metrics(batch))
    return k


def category_breakdown(batch: pd.DataFrame) -> pd.DataFrame:
    g = batch.assign(fraud_amount=batch["amount"] * batch["is_fraud"]).groupby("merchant_category").agg(
        txns=("is_fraud", "size"), frauds=("is_fraud", "sum"), fraud_amount=("fraud_amount", "sum"),
    )
    g["fraud_rate"] = g["frauds"] / g["txns"]
    return g.sort_values("fraud_rate", ascending=False)


def hourly_breakdown(batch: pd.DataFrame) -> pd.Series:
    fraud_hours = batch.loc[batch["is_fraud"] == 1, "hour_of_day"]
    return fraud_hours.value_counts().reindex(range(24), fill_value=0)


def pct(x):
    return f"{x:6.2%}"


def money(x):
    return f"${x:,.2f}"


def delta(cur, base, kind="rel"):
    if kind == "abs":
        return f"{(cur - base) * 100:+.2f} pp"
    if not base:
        return "n/a"
    return f"{(cur - base) / base:+.1%}"


def build_report(day: int, today: dict, base: dict, cats: pd.DataFrame, hours: pd.Series, trend: pd.DataFrame) -> str:
    W = 78
    lines = []
    add = lines.append
    rule = "=" * W
    sub = "-" * W
    report_date = SIM_START + timedelta(days=day - 1)

    add(rule)
    add(f"DAILY FRAUD MONITORING REPORT".center(W))
    add(f"Batch day {day} of {N_DAYS}  |  {report_date.isoformat()}  |  baseline: prior {BASELINE_DAYS} days".center(W))
    add(rule)

    # Alerts first so the reader sees problems immediately.
    alerts = []
    if base:
        if base["fraud_rate"] and (today["fraud_rate"] - base["fraud_rate"]) / base["fraud_rate"] > FRAUD_RATE_ALERT:
            alerts.append(f"Fraud rate {today['fraud_rate']:.2%} is >{FRAUD_RATE_ALERT:.0%} above baseline {base['fraud_rate']:.2%}")
        if base["precision"] - today["precision"] > PRECISION_DROP_ALERT:
            alerts.append(f"Precision drift: {today['precision']:.2%} vs baseline {base['precision']:.2%}")
        if base["recall"] - today["recall"] > RECALL_DROP_ALERT:
            alerts.append(f"Recall drift: {today['recall']:.2%} vs baseline {base['recall']:.2%}")
    add("\nALERTS")
    add(sub)
    if alerts:
        for a in alerts:
            add(f"  [!] {a}")
    else:
        add("  [ok] No KPI breached its alert threshold.")

    def section(title, rows):
        add(f"\n{title}")
        add(sub)
        add(f"  {'KPI':<36}{'Today':>14}{'Baseline':>14}{'Change':>12}")
        for label, key, fmt, kind in rows:
            cur = today[key]
            if base:
                add(f"  {label:<36}{fmt(cur):>14}{fmt(base[key]):>14}{delta(cur, base[key], kind):>12}")
            else:
                add(f"  {label:<36}{fmt(cur):>14}{'-':>14}{'-':>12}")

    count = lambda x: f"{x:,.0f}"
    hrs = lambda x: f"{x:,.2f} h"
    km = lambda x: f"{x:,.1f} km"

    section("1. VOLUME & FRAUD", [
        ("Transactions", "volume", count, "rel"),
        ("Total amount", "total_amount", money, "rel"),
        ("Fraud cases", "fraud_count", count, "rel"),
        ("Daily fraud rate", "fraud_rate", pct, "abs"),
        ("Fraud amount", "fraud_amount", money, "rel"),
        ("Fraud share of total $", "fraud_amount_share", pct, "abs"),
    ])
    section("2. TRANSACTION AMOUNTS", [
        ("Avg fraud amount", "avg_fraud_amount", money, "rel"),
        ("Avg legit amount", "avg_legit_amount", money, "rel"),
        ("Amount P50", "p50_amount", money, "rel"),
        ("Amount P90", "p90_amount", money, "rel"),
        ("Amount P99", "p99_amount", money, "rel"),
        ("Fraud amount P90", "p90_fraud_amount", money, "rel"),
        (f"Fraud rate when amount >{HIGH_RATIO:.0f}x 30d avg", "high_ratio_fraud_rate", pct, "abs"),
    ])
    section("3. GEOGRAPHY, VELOCITY & TIME", [
        ("Foreign txn share", "foreign_share", pct, "abs"),
        ("Foreign txn fraud rate", "foreign_fraud_rate", pct, "abs"),
        ("Domestic txn fraud rate", "domestic_fraud_rate", pct, "abs"),
        ("Median distance (fraud)", "median_distance_fraud", km, "rel"),
        ("Median distance (legit)", "median_distance_legit", km, "rel"),
        (f"Velocity: share with gap <{VELOCITY_GAP_HOURS:.0f}h", "velocity_share", pct, "abs"),
        (f"Velocity: fraud rate gap <{VELOCITY_GAP_HOURS:.0f}h", "velocity_fraud_rate", pct, "abs"),
        ("Median gap since last txn (fraud)", "median_gap_fraud", hrs, "rel"),
        ("Median gap since last txn (legit)", "median_gap_legit", hrs, "rel"),
        ("Night (00-05h) fraud rate", "night_fraud_rate", pct, "abs"),
        ("Share of fraud at night", "night_fraud_share", pct, "abs"),
    ])
    section("4. MODEL PERFORMANCE (gradient boosting, threshold 0.5)", [
        ("Alerts raised", "alerts", count, "rel"),
        ("True positives", "tp", count, "rel"),
        ("False positives", "fp", count, "rel"),
        ("Missed fraud (false negatives)", "fn", count, "rel"),
        ("Precision", "precision", pct, "abs"),
        ("Recall", "recall", pct, "abs"),
        ("F1 score", "f1", pct, "abs"),
        ("False positive rate", "fpr", pct, "abs"),
        ("Avg fraud score", "avg_fraud_score", pct, "abs"),
    ])

    add("\n5. FRAUD BY MERCHANT CATEGORY")
    add(sub)
    add(f"  {'Category':<16}{'Txns':>8}{'Frauds':>8}{'Rate':>10}{'Fraud $':>14}")
    for name, row in cats.iterrows():
        add(f"  {name:<16}{row['txns']:>8,.0f}{row['frauds']:>8,.0f}{row['fraud_rate']:>10.2%}{money(row['fraud_amount']):>14}")

    add("\n6. HOURLY FRAUD DISTRIBUTION (fraud cases per hour)")
    add(sub)
    peak = max(hours.max(), 1)
    for hour, n in hours.items():
        bar = "#" * int(round(n / peak * 40))
        add(f"  {hour:02d}:00  {n:>4}  {bar}")

    add(f"\n7. {len(trend)}-DAY TREND")
    add(sub)
    add(f"  {'Day':>4}{'Volume':>9}{'Frauds':>8}{'Fraud rate':>12}{'Precision':>11}{'Recall':>9}{'FPR':>9}")
    for d, row in trend.iterrows():
        marker = " <" if d == day else ""
        add(f"  {d:>4}{row['volume']:>9,.0f}{row['fraud_count']:>8,.0f}{row['fraud_rate']:>12.2%}"
            f"{row['precision']:>11.2%}{row['recall']:>9.2%}{row['fpr']:>9.3%}{marker}")
    add(f"\n  Precision drift (today - {BASELINE_DAYS}d baseline): "
        + (f"{(today['precision'] - base['precision']) * 100:+.2f} pp" if base else "n/a"))
    add(f"  Precision std dev across all days: {trend['precision'].std() * 100:.2f} pp")
    add(rule)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Daily batch fraud monitoring report")
    parser.add_argument("--day", type=int, default=N_DAYS, help=f"day to report on (1-{N_DAYS})")
    args = parser.parse_args()
    if not 1 <= args.day <= N_DAYS:
        parser.error(f"--day must be between 1 and {N_DAYS}")

    df = load_data()
    bundle = load_model()
    df["score"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    df["predicted"] = (df["score"] >= THRESHOLD).astype(int)
    df["day"] = assign_days(df)

    daily = {d: compute_kpis(batch) for d, batch in df.groupby("day")}
    trend = pd.DataFrame.from_dict(daily, orient="index")

    today = daily[args.day]
    base_days = [d for d in range(args.day - BASELINE_DAYS, args.day) if d >= 1]
    base = trend.loc[base_days].mean().to_dict() if base_days else {}

    batch = df[df["day"] == args.day]
    report = build_report(args.day, today, base, category_breakdown(batch), hourly_breakdown(batch), trend)

    print(report)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "daily_report.txt"
    out.write_text(report + "\n")
    print(f"\nSaved report to {out}")


if __name__ == "__main__":
    main()
