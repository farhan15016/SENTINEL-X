
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

EXCLUDED_COLUMNS = ["id", "attack_cat", "label"]
RANDOM_STATE = 42

CONTAMINATION_VALUES = [0.01, 0.05, 0.10, 0.20]


# ============================================================
# PIPELINE BUILDER
# ============================================================

def build_pipeline(
    X_normal: pd.DataFrame,
    feature_columns: list[str],
    contamination: float,
) -> Pipeline:
    """Build an Isolation Forest pipeline using normal traffic."""

    categorical_columns = X_normal.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_columns),
        ("numerical", numerical_pipeline, numerical_columns),
    ])

    detector = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("detector", detector),
    ])


# ============================================================
# EVALUATION HELPER
# ============================================================

def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Evaluate attack detection. Label 1 means attack."""

    raw_predictions = model.predict(X)
    predictions = (raw_predictions == -1).astype(int)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    return {
        "records": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "attack_precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "attack_recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "attack_f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    start_time = time.time()

    print("=" * 65)
    print("SENTINEL-X ISOLATION FOREST THRESHOLD EXPERIMENT")
    print("=" * 65, flush=True)

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...", flush=True)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(
            "Training and testing dataset schemas do not match."
        )

    feature_columns = [
        column for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df["label"].astype(int)

    if not y_train.isin([0, 1]).all():
        raise ValueError("Training labels must be 0 or 1.")

    if not y_test.isin([0, 1]).all():
        raise ValueError("Testing labels must be 0 or 1.")

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records:  {len(X_test):,}")

    # --------------------------------------------------------
    # 2. Split training data for contamination selection
    # --------------------------------------------------------

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_fit_normal = X_fit.loc[y_fit == 0].copy()

    print(f"Validation records: {len(X_val):,}")
    print(
        "Normal records used during threshold selection:",
        f"{len(X_fit_normal):,}",
    )

    # --------------------------------------------------------
    # 3. Compare contamination values on validation data
    # --------------------------------------------------------

    validation_results = []

    for contamination in CONTAMINATION_VALUES:
        print(
            f"\nTraining validation model: "
            f"contamination={contamination:.2f}",
            flush=True,
        )

        pipeline = build_pipeline(
            X_fit_normal,
            feature_columns,
            contamination,
        )

        pipeline.fit(X_fit_normal)

        metrics = evaluate_model(
            pipeline,
            X_val,
            y_val,
        )

        result = {
            "contamination": contamination,
            **metrics,
        }

        validation_results.append(result)

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['attack_precision']:.4f}\n"
            f"Recall:    {metrics['attack_recall']:.4f}\n"
            f"F1-score:  {metrics['attack_f1']:.4f}",
            flush=True,
        )

    # Select using validation F1, not test-set results.
    validation_results.sort(
        key=lambda item: item["attack_f1"],
        reverse=True,
    )

    best_result = validation_results[0]
    best_contamination = best_result["contamination"]

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS — RANKED BY ATTACK F1")
    print("=" * 65)

    for result in validation_results:
        print(
            f"Contamination={result['contamination']:.2f} | "
            f"F1={result['attack_f1']:.4f} | "
            f"Recall={result['attack_recall']:.4f} | "
            f"Precision={result['attack_precision']:.4f}"
        )

    print(
        f"\nSelected contamination: {best_contamination:.2f}"
    )

    # --------------------------------------------------------
    # 4. Refit selected model on all normal training records
    # --------------------------------------------------------

    X_all_normal = X_train.loc[y_train == 0].copy()

    print(
        f"\nRefitting selected model on "
        f"{len(X_all_normal):,} normal training records...",
        flush=True,
    )

    final_model = build_pipeline(
        X_all_normal,
        feature_columns,
        best_contamination,
    )

    final_model.fit(X_all_normal)

    # --------------------------------------------------------
    # 5. Evaluate once on the official testing dataset
    # --------------------------------------------------------

    print("\nEvaluating on the official testing dataset...", flush=True)

    test_metrics = evaluate_model(
        final_model,
        X_test,
        y_test,
    )

    print("\n===== OFFICIAL TEST-SET RESULTS =====")
    print(f"Contamination: {best_contamination:.2f}")
    print(f"Records:       {test_metrics['records']:,}")
    print(f"Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Attack precision: {test_metrics['attack_precision']:.4f}")
    print(f"Attack recall:    {test_metrics['attack_recall']:.4f}")
    print(f"Attack F1-score:  {test_metrics['attack_f1']:.4f}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    for row in test_metrics["confusion_matrix"]:
        print(row)

    print("\nClassification report:")
    report = test_metrics["classification_report"]

    print(
        f"{'Class':<12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
    )

    for label in ["Normal", "Attack"]:
        row = report[label]
        print(
            f"{label:<12}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>12.4f}"
            f"{row['f1-score']:>12.4f}"
        )

    # --------------------------------------------------------
    # 6. Save results separately; do not overwrite production
    # --------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Save the selected model separately for API integration.
    tuned_model_path = MODEL_DIR / "isolation_forest_tuned.joblib"
    joblib.dump(final_model, tuned_model_path)
    print(f"Tuned Isolation Forest saved to: {tuned_model_path}")
    validation_path = (
        MODEL_DIR / "isolation_forest_threshold_experiment.json"
    )

    test_results_path = (
        MODEL_DIR / "isolation_forest_threshold_test_results.json"
    )

    with validation_path.open("w", encoding="utf-8") as file:
        json.dump(validation_results, file, indent=2)

    test_results = {
        "model": "IsolationForest",
        "selected_contamination": best_contamination,
        "selection_metric": "validation_attack_f1",
        "training_strategy": "fit_on_normal_training_records_only",
        "validation_results": validation_results,
        "official_test_results": test_metrics,
        "random_state": RANDOM_STATE,
        "test_dataset": TEST_FILE.name,
        "production_model_overwritten": False,
    }

    with test_results_path.open("w", encoding="utf-8") as file:
        json.dump(test_results, file, indent=2)

    print("\n===== SAVED FILES =====")
    print(f"Validation results: {validation_path}")
    print(f"Test results:       {test_results_path}")
    print(
        "\nExisting isolation_forest.joblib and "
        "isolation_forest_metrics.json were not changed."
    )

    print(f"Elapsed time: {time.time() - start_time:.1f} seconds")
    print("\nTHRESHOLD EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
