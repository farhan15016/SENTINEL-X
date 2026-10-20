
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "random_forest.joblib"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

EXCLUDED_COLUMNS = ["id", "attack_cat", "label"]


def main():
    print("=" * 50)
    print("SENTINEL-X SAVED MODEL TEST")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

    print("\nLoading saved model...")
    model = joblib.load(MODEL_PATH)

    print("Loading test data...")
    df = pd.read_csv(TEST_FILE)

    if "label" not in df.columns:
        raise ValueError("The test dataset has no 'label' column.")

    feature_columns = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    X_test = df[feature_columns]
    y_test = df["label"].astype(int)

    print("\nGenerating predictions...")
    predictions = model.predict(X_test)

    print(f"Predictions generated: {len(predictions):,}")
    print(f"Correct predictions: {(predictions == y_test).sum():,}")
    print(f"Accuracy: {(predictions == y_test).mean():.4%}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nSaved model test completed successfully.")


if __name__ == "__main__":
    main()
