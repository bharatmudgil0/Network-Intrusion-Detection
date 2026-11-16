
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading final NIDS model...")

# Load complete pipeline
bundle = joblib.load("models/final_nids_model.pkl")

scaler = bundle["scaler"]
pca = bundle["pca"]
model = bundle["model"]
feature_names = bundle["feature_names"]

print("Final model loaded successfully!")

# Load original model dataset
print("\nLoading original dataset...")

df = pd.read_csv(
    "dataset/processed/model_dataset.csv"
)

# Separate features and labels
X = df[feature_names].copy()
y = df["Label"]

# Ensure numeric features
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinity with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Remove invalid rows
valid_rows = X.notna().all(axis=1)

X = X.loc[valid_rows]
y = y.loc[valid_rows]

# Recreate the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nOriginal test features:", X_test.shape)

# Apply the saved preprocessing steps
print("\nApplying StandardScaler...")

X_test_scaled = scaler.transform(X_test)

print("Applying PCA...")

X_test_pca = pca.transform(X_test_scaled)

print("PCA test shape:", X_test_pca.shape)

# Predict using Random Forest
print("\nPredicting network traffic...")

y_pred = model.predict(X_test_pca)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("FINAL NIDS PIPELINE PERFORMANCE")
print("=" * 55)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Compare with previously saved Random Forest predictions
saved_predictions = pd.read_csv(
    "results/random_forest_predictions.csv"
)

same_predictions = np.array_equal(
    np.asarray(y_pred),
    saved_predictions["Predicted"].to_numpy()
)

print("\nComparison with previous Random Forest predictions:")

if same_predictions:
    print("SUCCESS: All predictions match!")
else:
    matching = np.mean(
        np.asarray(y_pred) ==
        saved_predictions["Predicted"].to_numpy()
    )
    print(f"Matching predictions: {matching * 100:.2f}%")

print("\nFinal pipeline verification completed!")
