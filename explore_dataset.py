
import os
import pandas as pd

DATASET_PATH = "dataset/raw"

# Find CSV files
csv_files = [
    file for file in os.listdir(DATASET_PATH)
    if file.lower().endswith(".csv")
]

if not csv_files:
    print("No CSV files found. Check dataset/raw folder.")
    exit()

print("Available CSV files:", len(csv_files))

# Inspect the first CSV file
file_path = os.path.join(DATASET_PATH, csv_files[0])

print("\nInspecting:", csv_files[0])

# Read only the first 5 rows
df = pd.read_csv(file_path, nrows=5, low_memory=False)

print("\nDataset shape (sample):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
