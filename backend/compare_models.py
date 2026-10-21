
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"


def main():
    print("=" * 55)
    print("SENTINEL-X MODEL COMPARISON")
    print("=" * 55)

    print("\nLoading test dataset...")
    test_df = pd.read_csv(DATA_FILE)

    excluded = ["id", "attack_cat", "label"]
    feature_columns = [
        column for column in test_df.columns
        if column not in excluded
    ]

    X_test = test_df[feature_columns]
    y_test = test_df["label"].astype(int)

    print(f"Test records: {len(X_test):,}")

    models = {
        "Random Forest": (
            joblib.load(MODEL_DIR / "random_forest.joblib"),
            "supervised",
        ),
        "Isolation Forest": (
            joblib.load(MODEL_DIR / "isolation_forest.joblib"),
            "unsupervised",
        ),
    }

    results = {}

    for name, (model, approach) in models.items():
        print(f"\nEvaluating {name}...")

        if name == "Random Forest":
            predictions = model.predict(X_test).astype(int)
        else:
            raw_predictions = model.predict(X_test)
            predictions = (raw_predictions == -1).astype(int)

        metrics = {
            "approach": approach,
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "attack_precision": round(
                float(precision_score(y_test, predictions, zero_division=0)), 4
            ),
            "attack_recall": round(
                float(recall_score(y_test, predictions, zero_division=0)), 4
            ),
            "attack_f1": round(
                float(f1_score(y_test, predictions, zero_division=0)), 4
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["normal", "attack"],
                output_dict=True,
                zero_division=0,
            ),
        }

        results[name] = metrics

        print(f"Accuracy:         {metrics['accuracy']:.2%}")
        print(f"Attack precision: {metrics['attack_precision']:.2%}")
        print(f"Attack recall:    {metrics['attack_recall']:.2%}")
        print(f"Attack F1-score:  {metrics['attack_f1']:.2%}")
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

    report_path = MODEL_DIR / "model_comparison.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nComparison report saved to: {report_path}")
    print("\nMODEL COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
