
import pandas as pd

FILE_PATH = "dataset/processed/final_cicids2017.csv"

# Store label counts
label_counts = {}

total_records = 0

print("Reading final dataset labels...\n")

# Read only the Label column to save memory
for chunk in pd.read_csv(
    FILE_PATH,
    usecols=["Label"],
    chunksize=100000,
    low_memory=False
):

    # Normalize label whitespace
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    counts = chunk["Label"].value_counts()

    for label, count in counts.items():
        label_counts[label] = (
            label_counts.get(label, 0) + count
        )

    total_records += len(chunk)

# Convert results into a DataFrame
distribution = pd.DataFrame(
    list(label_counts.items()),
    columns=["Attack Type", "Count"]
)

# Calculate percentages
distribution["Percentage"] = (
    distribution["Count"] / total_records * 100
)

# Sort from most frequent to least frequent
distribution = distribution.sort_values(
    by="Count",
    ascending=False
)

print("=" * 65)
print("CLASS DISTRIBUTION")
print("=" * 65)

print(distribution.to_string(
    index=False,
    formatters={
        "Percentage": "{:.4f}%".format
    }
))

print("\nTotal records:", total_records)
print("Total classes:", len(distribution))
