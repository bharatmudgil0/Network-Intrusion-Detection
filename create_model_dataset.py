
import os
import pandas as pd

INPUT_FILE = "dataset/processed/final_cicids2017.csv"
OUTPUT_FILE = "dataset/processed/model_dataset.csv"

MAX_PER_CLASS = 50000
CHUNK_SIZE = 100000
RANDOM_STATE = 42

# Store a sample for each class
class_samples = {}

print("Creating a stratified dataset sample...\n")

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    # Normalize column names and labels
    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = (
        chunk["Label"]
        .astype(str)
        .str.strip()
    )

    # Make corrupted separator characters readable
    chunk["Label"] = chunk["Label"].str.replace(
        "\uFFFD", "-", regex=False
    )

    # Process each class separately
    for label, group in chunk.groupby("Label"):

        if label not in class_samples:
            class_samples[label] = group.copy()
        else:
            class_samples[label] = pd.concat(
                [class_samples[label], group],
                ignore_index=True
            )

        # Keep at most MAX_PER_CLASS records
        if len(class_samples[label]) > MAX_PER_CLASS:
            class_samples[label] = (
                class_samples[label]
                .sample(
                    n=MAX_PER_CLASS,
                    random_state=RANDOM_STATE
                )
                .reset_index(drop=True)
            )

    print("Processed a chunk...")

# Combine all class samples
model_df = pd.concat(
    class_samples.values(),
    ignore_index=True
)

# Shuffle records
model_df = model_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Save the model dataset
os.makedirs("dataset/processed", exist_ok=True)

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 50)
print("MODEL DATASET CREATED")
print("=" * 50)

print("Total records:", len(model_df))
print("Total features:", len(model_df.columns) - 1)
print("Total classes:", model_df["Label"].nunique())

print("\nClass distribution:")
print(model_df["Label"].value_counts())

print("\nSaved to:", OUTPUT_FILE)
