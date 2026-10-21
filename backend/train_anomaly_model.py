
from pathlib import Path
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
)
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


def main():
    start_time = time.time()

    print("=" * 60)
    print("SENTINEL-X ISOLATION FOREST TRAINING")
    print("=" * 60, flush=True)

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------
    print("\nLoading datasets...", flush=True)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Training and testing schemas do not match.")

    feature_columns = [
        col for col in train_df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df["label"].astype(int)

    if not y_train.isin([0, 1]).all():
        raise ValueError("Training labels must be 0 or 1.")

    if not y_test.isin([0, 1]).all():
        raise ValueError("Testing labels must be 0 or 1.")

    # --------------------------------------------------------
    # 2. Select only normal traffic for fitting
    # --------------------------------------------------------
    X_normal = X_train.loc[y_train == 0].copy()

    if X_normal.empty:
        raise ValueError("No normal training records were found.")

    print(f"Training records available: {len(X_train):,}")
    print(f"Normal records used for fitting: {len(X_normal):,}")
    print(f"Testing records: {len(X_test):,}")

    # --------------------------------------------------------
    # 3. Identify feature types
    # --------------------------------------------------------
    categorical_columns = X_normal.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_columns = [
        col for col in feature_columns
        if col not in categorical_columns
    ]

    print(f"Categorical features: {categorical_columns}")
    print(f"Numerical features: {len(numerical_columns)}")

    # --------------------------------------------------------
    # 4. Build preprocessing pipeline
    # --------------------------------------------------------
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_columns),
        ("numerical", numerical_pipeline, numerical_columns),
    ])

    # --------------------------------------------------------
    # 5. Configure Isolation Forest
    # --------------------------------------------------------
    # Fit on normal traffic only. Use a conservative contamination
    # setting because the training examples are selected as normal.
    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("detector", anomaly_detector),
    ])

    # --------------------------------------------------------
    # 6. Fit on normal traffic only
    # --------------------------------------------------------
    print("\nTraining Isolation Forest...", flush=True)
    pipeline.fit(X_normal)

    # --------------------------------------------------------
    # 7. Evaluate on the separate test dataset
    # --------------------------------------------------------
    print("\nEvaluating on testing data...", flush=True)

    # Isolation Forest: 1 = inlier, -1 = outlier.
    raw_predictions = pipeline.predict(X_test)

    # Convert to dataset labels: 0 = normal, 1 = attack.
    predictions = (raw_predictions == -1).astype(int)

    # Higher decision_function values are more normal.
    # Negating the value gives an anomaly score where higher
    # values indicate more anomalous behaviour.
    anomaly_scores = -pipeline.decision_function(X_test)

    print(
        "\nAccuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ))

    print("\nAnomaly score summary:")
    print(f"Minimum: {anomaly_scores.min():.4f}")
    print(f"Maximum: {anomaly_scores.max():.4f}")
    print(f"Mean:    {anomaly_scores.mean():.4f}")

    # --------------------------------------------------------
    # 8. Save model and evaluation report
    # --------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "isolation_forest.joblib"
    joblib.dump(pipeline, model_path)

    results = {
        "model": "IsolationForest",
        "training_strategy": "normal_training_records_only",
        "training_normal_records": int(len(X_normal)),
        "testing_records": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
        "contamination": 0.01,
        "random_state": RANDOM_STATE,
    }

    results_path = MODEL_DIR / "isolation_forest_metrics.json"

    with results_path.open("w", encoding="utf-8") as file:
        import json
        json.dump(results, file, indent=2)

    elapsed = time.time() - start_time

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {results_path}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print("\nISOLATION FOREST TRAINING COMPLETE")


if __name__ == "__main__":
    main()
