
import os

# Dataset folder
DATASET_PATH = "dataset/raw"

# Get all CSV files
csv_files = [
    file for file in os.listdir(DATASET_PATH)
    if file.lower().endswith(".csv")
]

print("Total CSV files found:", len(csv_files))
print("\nDataset files:")

for file in csv_files:
    file_path = os.path.join(DATASET_PATH, file)

    # File size in MB
    size = os.path.getsize(file_path) / (1024 * 1024)

    print(f"{file} - {size:.2f} MB")
