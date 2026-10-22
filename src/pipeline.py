"""Train and evaluate fraud detection models.

- Preprocessing: standard scaling for numeric features, one-hot encoding for merchant category.
- Isolation Forest: unsupervised anomaly detection (trained without labels).
- GradientBoostingClassifier: supervised classifier, saved to model.pkl with its preprocessing.

Usage:
    python src/pipeline.py
"""

import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES, TARGET, load_data,
)

SEED = 42
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])


def fraud_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name} (fraud class)")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"  {key:<10} {value:.4f}")


def train_isolation_forest(X_train, X_test, y_test, contamination: float):
    model = Pipeline([
        ("prep", build_preprocessor()),
        ("iso", IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1)),
    ])
    model.fit(X_train)  # unsupervised: labels are not used
    # IsolationForest returns -1 for anomalies and 1 for inliers.
    y_pred = (model.predict(X_test) == -1).astype(int)
    metrics = fraud_metrics(y_test, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_test, -model.decision_function(X_test))
    return model, metrics


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ("prep", build_preprocessor()),
        ("gb", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=SEED,
        )),
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = fraud_metrics(y_test, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    return model, metrics, y_pred


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED,
    )
    print(f"Loaded {len(df):,} transactions | fraud rate {y.mean():.2%}")
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    _, iso_metrics = train_isolation_forest(X_train, X_test, y_test, contamination=float(y_train.mean()))
    print_metrics("Isolation Forest", iso_metrics)

    gb_model, gb_metrics, gb_pred = train_gradient_boosting(X_train, y_train, X_test, y_test)
    print_metrics("Gradient Boosting", gb_metrics)
    print("\nGradient Boosting classification report:")
    print(classification_report(y_test, gb_pred, target_names=["legit", "fraud"], digits=4))

    # Top features by importance, mapped back to readable names.
    names = gb_model.named_steps["prep"].get_feature_names_out()
    importances = gb_model.named_steps["gb"].feature_importances_
    print("Top 5 features:")
    for idx in np.argsort(importances)[::-1][:5]:
        print(f"  {names[idx]:<45} {importances[idx]:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": gb_model, "features": FEATURES, "metrics": gb_metrics}, f)
    print(f"\nSaved gradient boosting model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
