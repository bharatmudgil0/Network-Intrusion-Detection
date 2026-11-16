
import os
import pandas as pd
import numpy as np

DATASET_PATH = "dataset/raw"

csv_files = [
    file for file in os.listdir(DATASET_PATH)
    if file.lower().endswith(".csv")
]

print("Starting dataset inspection...\n")

for file in csv_files:

    file_path = os.path.join(DATASET_PATH, file)

    print("=" * 70)
    print("FILE:", file)

    total_rows = 0
    missing_values = {}
    infinite_values = {}
    label_counts = {}

    # Read the CSV in smaller chunks
    for chunk in pd.read_csv(
        file_path,
        chunksize=100000,
        low_memory=False
    ):

        total_rows += len(chunk)

        # Remove extra spaces from column names
        chunk.columns = chunk.columns.str.strip()

        # Count missing values
        missing = chunk.isnull().sum()

        for column, count in missing.items():
            missing_values[column] = (
                missing_values.get(column, 0) + count
            )

        # Count infinite numeric values
        numeric_data = chunk.select_dtypes(include=np.number)

        inf_counts = np.isinf(numeric_data).sum()

        for column, count in inf_counts.items():
            infinite_values[column] = (
                infinite_values.get(column, 0) + count
            )

        # Count target labels
        if "Label" in chunk.columns:
            labels = chunk["Label"].astype(str).str.strip()
            counts = labels.value_counts()

            for label, count in counts.items():
                label_counts[label] = (
                    label_counts.get(label, 0) + count
                )

    print("\nTotal rows:", total_rows)

    print("\nMissing values:")
    print({
        key: value
        for key, value in missing_values.items()
        if value > 0
    })

    print("\nInfinite values:")
    print({
        key: value
        for key, value in infinite_values.items()
        if value > 0
    })

    print("\nAttack label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")

print("\nDataset inspection completed!")
