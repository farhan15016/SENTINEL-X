
from pathlib import Path
import json

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "random_forest.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "models" / "feature_importance.json"


def main():
    print("=" * 55)
    print("SENTINEL-X FEATURE IMPORTANCE ANALYSIS")
    print("=" * 55)

    print("\nLoading trained Random Forest...")
    model = joblib.load(MODEL_PATH)

    print("Loading test dataset...")
    test_df = pd.read_csv(DATA_PATH)

    excluded = ["id", "attack_cat", "label"]
    feature_columns = [
        column for column in test_df.columns
        if column not in excluded
    ]

    # The trained pipeline preserves the original input feature names.
    if hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    X_test = test_df[feature_columns]
    y_test = test_df["label"].astype(int)

    # Calculate permutation importance on a manageable sample.
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    print(f"\nAnalyzing {sample_size:,} test records...")
    print("Calculating permutation importance...")

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="f1",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        "importance_mean", ascending=False
    )

    print("\nTop 15 features:")
    print("-" * 55)

    for _, row in importance_df.head(15).iterrows():
        print(
            f"{row['feature']:<25} "
            f"{row['importance_mean']:+.5f} "
            f"+/- {row['importance_std']:.5f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "method": "permutation_importance",
        "scoring": "f1",
        "sample_size": sample_size,
        "n_repeats": 3,
        "features": [
            {
                "feature": str(row["feature"]),
                "importance_mean": round(float(row["importance_mean"]), 6),
                "importance_std": round(float(row["importance_std"]), 6),
            }
            for _, row in importance_df.iterrows()
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved feature report: {OUTPUT_PATH}")
    print("\nFEATURE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
