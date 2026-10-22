"""Generate fraud analytics charts as PNGs in output/.

Charts:
    fraud_by_category.png      fraud rate per merchant category
    hourly_distribution.png    share of fraud vs. legit transactions by hour
    amount_distribution.png    transaction amount distribution, fraud vs. legit
    precision_recall_curve.png precision-recall curves for both models on the test split

Usage:
    python src/visualizations.py
"""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from common import MODEL_PATH, OUTPUT_DIR, TARGET, load_data
from pipeline import FEATURES, SEED, train_isolation_forest

# Palette: categorical slots 1-2 (blue, orange) on a light surface.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
LEGIT = "#2a78d6"
FRAUD = "#eb6834"


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "legend.frameon": False,
        "font.size": 11,
    })


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_fraud_by_category(df):
    stats = df.groupby("merchant_category")[TARGET].agg(["mean", "sum"]).sort_values("mean")
    overall = df[TARGET].mean()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(stats.index, stats["mean"] * 100, color=LEGIT, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.axvline(overall * 100, color=TEXT_MUTED, linestyle="--", linewidth=1, zorder=0)
    ax.text(overall * 100, len(stats) - 0.4, f" overall {overall:.1%}", color=TEXT_MUTED, fontsize=9, va="bottom")
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{row['mean']:.1%}  ({int(row['sum']):,})", va="center", color=TEXT, fontsize=9,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.set_xlabel("Fraud rate (%)  ·  label shows rate and fraud count")
    ax.set_xlim(0, stats["mean"].max() * 100 * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_title("Fraud rate by merchant category")
    save(fig, "fraud_by_category.png")


def plot_hourly_distribution(df):
    # Share of each class's transactions per hour, so both fit one axis.
    share = (df.groupby([TARGET, "hour_of_day"]).size()
             .groupby(level=0).transform(lambda s: s / s.sum() * 100).unstack(0))
    hours = share.index

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hours, share[0], color=LEGIT, linewidth=2, marker="o", markersize=5, label="Legitimate")
    ax.plot(hours, share[1], color=FRAUD, linewidth=2, marker="o", markersize=5, label="Fraud")
    ax.text(hours[-1] + 0.3, share[0].iloc[-1], "Legitimate", color=TEXT, va="center", fontsize=9)
    ax.text(hours[-1] + 0.3, share[1].iloc[-1], "Fraud", color=TEXT, va="center", fontsize=9)
    ax.axvspan(-0.5, 5.5, color=GRID, alpha=0.5, linewidth=0)
    ax.text(5.35, ax.get_ylim()[1] * 0.98, "night (00-05h)", ha="right", va="top", color=TEXT_MUTED, fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlim(-0.5, 25)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of class transactions (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("When transactions happen: fraud vs. legitimate", pad=34)
    save(fig, "hourly_distribution.png")


def plot_amount_distribution(df):
    bins = np.logspace(np.log10(0.5), np.log10(df["amount"].max()), 60)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, value, color in [("Legitimate", 0, LEGIT), ("Fraud", 1, FRAUD)]:
        amounts = df.loc[df[TARGET] == value, "amount"]
        weights = np.full(len(amounts), 100 / len(amounts))  # each class sums to 100%
        ax.hist(amounts, bins=bins, weights=weights, histtype="stepfilled", alpha=0.25, color=color)
        ax.hist(amounts, bins=bins, weights=weights, histtype="step", linewidth=2, color=color,
                label=f"{label}  (median ${amounts.median():,.0f})")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Share of class transactions (%)")
    ax.legend(loc="upper right")
    ax.set_title("Transaction amount distribution")
    save(fig, "amount_distribution.png")


def plot_precision_recall(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    with open(MODEL_PATH, "rb") as f:
        gb = pickle.load(f)["model"]
    gb_scores = gb.predict_proba(X_test)[:, 1]
    iso, _ = train_isolation_forest(X_train, X_test, y_test, contamination=float(y_train.mean()))
    iso_scores = -iso.decision_function(X_test)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, scores, color in [("Gradient Boosting", gb_scores, LEGIT), ("Isolation Forest", iso_scores, FRAUD)]:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{name}  (AP = {ap:.3f})")
    ax.axhline(y_test.mean(), color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax.text(0.3, y_test.mean() + 0.015, f"random baseline ({y_test.mean():.1%})", color=TEXT_MUTED, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.45))
    ax.set_title("Precision-recall curve (test set)")
    save(fig, "precision_recall_curve.png")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_style()
    df = load_data()
    plot_fraud_by_category(df)
    plot_hourly_distribution(df)
    plot_amount_distribution(df)
    if MODEL_PATH.exists():
        plot_precision_recall(df)
    else:
        print(f"Skipping precision_recall_curve.png: {MODEL_PATH} not found. Run `python src/pipeline.py` first.")


if __name__ == "__main__":
    main()
