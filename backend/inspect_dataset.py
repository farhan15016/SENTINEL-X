
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
]

for filename in FILES:
    filepath = DATA_DIR / filename

    print("\n" + "=" * 55)
    print(f"DATASET: {filename}")
    print("=" * 55)

    if not filepath.exists():
        print("ERROR: File not found:", filepath)
        continue

    with filepath.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        if not columns:
            print("ERROR: CSV headers are missing.")
            continue

        row_count = 0
        missing_counts = {column: 0 for column in columns}
        label_counts = {}
        attack_counts = {}
        first_row = None

        for row in reader:
            row_count += 1

            if first_row is None:
                first_row = row

            for column in columns:
                value = row.get(column)
                if value is None or value.strip() == "":
                    missing_counts[column] += 1

            label = row.get("label")
            if label is not None:
                label = label.strip()
                label_counts[label] = label_counts.get(label, 0) + 1

            attack = row.get("attack_cat")
            if attack is not None:
                attack = attack.strip() or "(blank)"
                attack_counts[attack] = attack_counts.get(attack, 0) + 1

    print("Rows:", row_count)
    print("Columns:", len(columns))
    print("\nColumn names:")
    print(", ".join(columns))

    print("\nLabel distribution:")
    print(label_counts or "No 'label' column found")

    print("\nAttack categories (top 15):")
    for category, count in sorted(
        attack_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:15]:
        print(f"  {category}: {count}")

    missing = {
        column: count
        for column, count in missing_counts.items()
        if count > 0
    }

    print("\nColumns with blank values:")
    print(missing if missing else "None found")

print("\n" + "=" * 55)
print("DUPLICATE ROW CHECK")
print("=" * 55)

for filename in FILES:
    filepath = DATA_DIR / filename

    seen_rows = set()
    duplicates = 0

    with filepath.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)

        for row in reader:
            row_key = tuple(row)

            if row_key in seen_rows:
                duplicates += 1
            else:
                seen_rows.add(row_key)

    print(f"{filename}: {duplicates} duplicate rows")

print("\n" + "=" * 55)
print("FINAL DATASET VALIDATION")
print("=" * 55)

training_file = DATA_DIR / "UNSW_NB15_training-set.csv"
testing_file = DATA_DIR / "UNSW_NB15_testing-set.csv"

with training_file.open("r", encoding="utf-8-sig", newline="") as file:
    training_columns = next(csv.reader(file))

with testing_file.open("r", encoding="utf-8-sig", newline="") as file:
    testing_columns = next(csv.reader(file))

print("Training columns:", len(training_columns))
print("Testing columns:", len(testing_columns))
print("Same columns in same order:", training_columns == testing_columns)

excluded_columns = ["id", "label", "attack_cat"]
model_features = [
    column for column in training_columns
    if column not in excluded_columns
]

print("\nTarget column: label")
print("Excluded from model inputs:", excluded_columns)
print("Candidate model features:", len(model_features))
print(", ".join(model_features))

if training_columns != testing_columns:
    print("\nWARNING: Dataset schemas differ.")
else:
    print("\nSchema validation passed.")
