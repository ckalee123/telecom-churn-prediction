"""
Telecom Customer Churn Prediction
---------------------------------
Predicts which telecom customers are likely to leave (churn), using the
public IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

Steps:
  1. Load and clean the data
  2. Feature engineering
  3. Train and compare 3 models with stratified 5-fold cross-validation
  4. Evaluate the best model on a held-out test set
  5. Explain the model (feature importance) and measure business value (lift)

Run:  python churn_prediction.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, classification_report, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("data/Telco-Customer-Churn.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)


# 1. Load and clean ---------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges is stored as text and is blank for brand-new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


# 2. Feature engineering ----------------------------------------------------
SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # How many add-on services the customer uses
    df["num_services"] = (df[SERVICES] == "Yes").sum(axis=1)
    # Average monthly spend over the customer's lifetime
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0,
                                       df["TotalCharges"] / df["tenure"],
                                       df["MonthlyCharges"])
    # Tenure buckets (new, 1-2 years, 2-4 years, loyal)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12m", "12-24m", "24-48m", "48-72m"]).astype(str)
    # Customer on a month-to-month contract AND without tech support: a known high-risk profile
    df["mtm_no_support"] = ((df["Contract"] == "Month-to-month")
                            & (df["TechSupport"] == "No")).astype(int)
    return df


# 3. Models -----------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=3, random_state=RANDOM_STATE),
    }


def lift_at_top(y_true, y_score, top=0.10) -> float:
    """How many times more churners we find in the top-X% riskiest customers vs. random."""
    order = np.argsort(-y_score)
    n_top = int(len(y_true) * top)
    return np.asarray(y_true)[order[:n_top]].mean() / np.mean(y_true)


def main():
    df = add_features(load_data(DATA_PATH))
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    print(f"Customers: {len(df):,} | churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Compare models with stratified 5-fold cross-validation on the training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "f1"])
        rows.append({"model": name,
                     "cv_roc_auc": s["test_roc_auc"].mean(), "cv_roc_auc_std": s["test_roc_auc"].std(),
                     "cv_f1": s["test_f1"].mean()})
    cv_table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    cv_table.to_csv(OUT / "cv_results.csv", index=False)
    print("\nCross-validation (train set):\n", cv_table.round(3).to_string(index=False))

    # Evaluate all models on the held-out test set
    fig, ax = plt.subplots(figsize=(6, 5))
    test_rows, fitted = [], {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        # choose the decision threshold that maximizes F1 on out-of-fold TRAIN predictions
        # (never on the test set, to avoid leakage)
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        best_t = max(thresholds, key=lambda t: f1_score(y_train, oof >= t))
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        proba = pipe.predict_proba(X_test)[:, 1]
        test_rows.append({"model": name,
                          "test_roc_auc": roc_auc_score(y_test, proba),
                          "test_f1": f1_score(y_test, proba >= best_t),
                          "threshold": best_t,
                          "lift_top10%": lift_at_top(y_test, proba, 0.10)})
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curves - test set")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=150)

    test_table = pd.DataFrame(test_rows).sort_values("test_roc_auc", ascending=False)
    test_table.to_csv(OUT / "test_results.csv", index=False)
    print("\nHeld-out test set:\n", test_table.round(3).to_string(index=False))

    # Detailed report + feature importance for the best model
    best = test_table.iloc[0]
    best_pipe = fitted[best["model"]]
    proba = best_pipe.predict_proba(X_test)[:, 1]
    print(f"\nBest model: {best['model']}\n")
    print(classification_report(y_test, proba >= best["threshold"], target_names=["Stay", "Churn"]))

    imp = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE)
    importance = (pd.Series(imp.importances_mean, index=X_test.columns)
                  .sort_values(ascending=True).tail(10))
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot.barh(ax=ax, color="#1f3a5f")
    ax.set_title(f"Top 10 features - {best['model']}\n(permutation importance, drop in ROC-AUC)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    print("Top features:\n", importance.sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
