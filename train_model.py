from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "employee_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "quit"
NUMERIC_FEATURES = [
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company", "Work_accident",
    "promotion_last_5years"
]
CATEGORICAL_FEATURES = ["department", "salary"]
REQUIRED = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[REQUIRED].dropna().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    candidates = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    }

    rows = []
    fitted = {}
    for name, classifier in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", classifier)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_quit": float(precision_score(y_test, pred, zero_division=0)),
            "recall_quit": float(recall_score(y_test, pred, zero_division=0)),
            "f1_quit": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba))
        }
        rows.append(metrics)
        fitted[name] = pipe
        print(f"\n=== {name} ===")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pred, target_names=["Stayed", "Quit"], zero_division=0))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Stayed", "Quit"], ax=ax, colorbar=False
        )
        ax.set_title(f"{name} — Confusion Matrix")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=150)
        plt.close(fig)

    results = pd.DataFrame(rows).sort_values("f1_quit", ascending=False)
    results.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    print("\n=== Model comparison (sorted by F1 for quit class) ===")
    print(results.to_string(index=False))

    best_name = str(results.iloc[0]["model"])
    joblib.dump(fitted[best_name], MODEL_DIR / "employee_turnover_model.joblib")
    (MODEL_DIR / "model_info.json").write_text(
        json.dumps({"selected_model": best_name, "target": TARGET,
                    "positive_class": 1, "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES},
                   indent=2),
        encoding="utf-8"
    )
    print(f"\nSaved best model: {best_name}")
    print(f"Model path: {MODEL_DIR / 'employee_turnover_model.joblib'}")
    print(f"Metrics path: {REPORT_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
