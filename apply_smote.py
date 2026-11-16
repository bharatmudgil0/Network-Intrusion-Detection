
import os
import joblib

from imblearn.over_sampling import SMOTE
from collections import Counter

INPUT_FILE = "dataset/processed/splits/pca_data.pkl"
OUTPUT_FILE = "dataset/processed/splits/smote_data.pkl"

print("Loading PCA-transformed data...")

X_train, X_test, y_train, y_test = joblib.load(INPUT_FILE)

print("\nBefore SMOTE:")
print(Counter(y_train))

# Set target size for minority classes
sampling_strategy = {
    label: 5000
    for label, count in Counter(y_train).items()
    if count < 5000
}

print("\nSMOTE target sizes:")
print(sampling_strategy)

# Apply SMOTE only to training data
smote = SMOTE(
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=5
)

print("\nApplying SMOTE...")

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# Save balanced training data and untouched test data
joblib.dump(
    (X_train_smote, X_test, y_train_smote, y_test),
    OUTPUT_FILE,
    compress=3
)

print("\n" + "=" * 50)
print("SMOTE COMPLETED")
print("=" * 50)

print("Before SMOTE:", X_train.shape)
print("After SMOTE:", X_train_smote.shape)

print("\nClass distribution after SMOTE:")
print(Counter(y_train_smote))

print("\nTesting data remains unchanged:", X_test.shape)

print("\nSaved:", OUTPUT_FILE)
