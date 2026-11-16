
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

# File paths
INPUT_FILE = "dataset/processed/model_dataset.csv"
OUTPUT_DIR = "dataset/processed/splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Normalize column names
df.columns = df.columns.str.strip()

# Normalize labels
df["Label"] = df["Label"].astype(str).str.strip()

# Separate features and target
X = df.drop(columns=["Label"])
y = df["Label"]

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove any records that became invalid
valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("\nDataset information:")
print("Total records:", len(X))
print("Total features:", X.shape[1])
print("Total classes:", y.nunique())

print("\nSplitting dataset...")

# Stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save all four datasets
joblib.dump(
    (X_train, X_test, y_train, y_test),
    os.path.join(OUTPUT_DIR, "train_test_data.pkl"),
    compress=3
)

print("\n" + "=" * 50)
print("DATA SPLITTING COMPLETED")
print("=" * 50)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nSaved training and testing data successfully!")
