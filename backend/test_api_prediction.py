
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"
API_URL = "http://127.0.0.1:8000/api/predict"

EXCLUDED_COLUMNS = ["id", "attack_cat", "label"]


def main():
    print("SENTINEL-X API PREDICTION TEST")
    print("=" * 40)

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_FILE}"
        )

    print("Loading one genuine test record...")
    df = pd.read_csv(TEST_FILE, nrows=1)

    feature_columns = [
        column for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    record = df[feature_columns].iloc[0].to_dict()

    # Convert NumPy scalar values to ordinary Python numbers.
    record = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in record.items()
    }

    print(f"Features supplied: {len(record)}")
    print("Sending record to the Random Forest API...")

    request = Request(
        API_URL,
        data=json.dumps(record).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            print(f"HTTP status: {response.status}")
            print(json.dumps(result, indent=2))

    except HTTPError as error:
        print(f"HTTP error: {error.code}")
        print(error.read().decode("utf-8"))

    except Exception as error:
        print(f"Test failed: {error}")


if __name__ == "__main__":
    main()
