
from pathlib import Path

import pandas as pd


# Resolve paths from the SENTINEL-X project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

TARGET_COLUMN = "label"
EXCLUDED_COLUMNS = ["id", "label", "attack_cat"]


def load_dataset(file_path):
    """Load a CSV and validate its basic structure."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Check the filename and make sure the CSV is inside data/."
        )

    df = pd.read_csv(file_path, low_memory=False)

    if df.empty:
        raise ValueError(f"The dataset is empty: {file_path.name}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COLUMN}' "
            f"in {file_path.name}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Missing labels detected in {file_path.name}"
        )

    return df


def main():
    print("=" * 55)
    print("SENTINEL-X DATASET PREPARATION")
    print("=" * 55)

    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)

    # Both files must have the same columns in the same order.
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(
            "Training and testing columns do not match."
        )

    # Exclude identifiers and the attack-category label from inputs.
    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(int)

    if not y_train.isin([0, 1]).all():
        raise ValueError("Training labels must be 0 or 1.")

    if not y_test.isin([0, 1]).all():
        raise ValueError("Testing labels must be 0 or 1.")

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Target column: {TARGET_COLUMN}")

    print("\nTraining labels:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nTesting labels:")
    print(y_test.value_counts().sort_index().to_string())

    print("\nMissing feature values:")
    print(f"Training: {int(X_train.isna().sum().sum()):,}")
    print(f"Testing:  {int(X_test.isna().sum().sum()):,}")

    print("\nFeature columns:")
    print(", ".join(feature_columns))

    print("\nDataset preparation checks passed.")


if __name__ == "__main__":
    main()
