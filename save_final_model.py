
import joblib
import pandas as pd
import os

print("Loading trained components...")

# Load fitted preprocessing components
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

# Load best trained classifier
random_forest = joblib.load("models/random_forest.pkl")

# Read original feature names in their exact training order
df_columns = pd.read_csv(
    "dataset/processed/model_dataset.csv",
    nrows=0
)

feature_names = [
    col for col in df_columns.columns
    if col != "Label"
]

# Create final model bundle
final_model = {
    "scaler": scaler,
    "pca": pca,
    "model": random_forest,
    "feature_names": feature_names
}

os.makedirs("models", exist_ok=True)

# Save everything together
joblib.dump(
    final_model,
    "models/final_nids_model.pkl"
)

print("\nFinal NIDS model saved successfully!")
print("File: models/final_nids_model.pkl")

print("\nNumber of input features:", len(feature_names))
print("PCA components:", pca.n_components_)
print("Classifier: Random Forest")
