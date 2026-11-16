
import os
import pandas as pd
import numpy as np

# Input and output folders
INPUT_PATH = "dataset/raw"
OUTPUT_PATH = "dataset/processed/cleaned_cicids2017.csv"

os.makedirs("dataset/processed", exist_ok=True)

# Find all CSV files
csv_files = [
    file for file in os.listdir(INPUT_PATH)
    if file.lower().endswith(".csv")
]

# Remove an old output file so reruns don't append duplicate data
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

total_original = 0
total_cleaned = 0
total_removed = 0
write_header = True

for file in csv_files:

    file_path = os.path.join(INPUT_PATH, file)

    print(f"\nProcessing: {file}")

    # Read data in chunks to reduce memory usage
    for chunk in pd.read_csv(
        file_path,
        chunksize=100000,
        low_memory=False
    ):

        original_rows = len(chunk)
        total_original += original_rows

        # 1. Remove extra spaces from column names
        chunk.columns = chunk.columns.str.strip()

        # 2. Clean whitespace around labels
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # 3. Replace positive and negative infinity with NaN
        numeric_columns = chunk.select_dtypes(
            include=np.number
        ).columns

        chunk[numeric_columns] = chunk[
            numeric_columns
        ].replace([np.inf, -np.inf], np.nan)

        # 4. Remove rows containing missing values
        chunk = chunk.dropna()

        cleaned_rows = len(chunk)

        total_cleaned += cleaned_rows
        total_removed += original_rows - cleaned_rows

        # 5. Save cleaned records
        chunk.to_csv(
            OUTPUT_PATH,
            mode="w" if write_header else "a",
            header=write_header,
            index=False
        )

        write_header = False

    print(f"Finished: {file}")

print("\n" + "=" * 50)
print("DATA CLEANING COMPLETED")
print("=" * 50)

print("Original records:", total_original)
print("Cleaned records:", total_cleaned)
print("Removed records:", total_removed)

if total_original > 0:
    print(
        "Records retained:",
        round(total_cleaned / total_original * 100, 2),
        "%"
    )

print("\nSaved dataset:", OUTPUT_PATH)
