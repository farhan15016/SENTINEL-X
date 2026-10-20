
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

EXCLUDED_COLUMNS = ["id", "attack_cat", "label"]


def main():
    print("=" * 55)
    print("SENTINEL-X RANDOM FOREST TRAINING")
    print("=" * 55, flush=True)

    start_time = time.time()

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

    categorical_columns = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_columns = [
        col for col in feature_columns
        if col not in categorical_columns
    ]

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Categorical features: {categorical_columns}")
    print(f"Numerical features: {len(numerical_columns)}")

    # Fit preprocessing only on the training data.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_columns),
        ("numerical", numerical_pipeline, numerical_columns),
    ])

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("\nTraining Random Forest...", flush=True)
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on the testing set...", flush=True)
    predictions = pipeline.predict(X_test)

    print("\nAccuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "random_forest.joblib"

    joblib.dump(pipeline, model_path)

    elapsed = time.time() - start_time

    print(f"\nModel saved to: {model_path}")
    print(f"Training and evaluation time: {elapsed:.1f} seconds")
    print("\nSENTINEL-X TRAINING COMPLETE")


if __name__ == "__main__":
    main()
