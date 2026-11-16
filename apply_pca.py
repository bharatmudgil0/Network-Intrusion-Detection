
import os
import joblib

from sklearn.decomposition import PCA

# File paths
INPUT_FILE = "dataset/processed/splits/scaled_data.pkl"
OUTPUT_FILE = "dataset/processed/splits/pca_data.pkl"

os.makedirs("models", exist_ok=True)

print("Loading scaled dataset...")

X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(
    INPUT_FILE
)

print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

# Retain 95% of the variance
pca = PCA(
    n_components=0.95,
    svd_solver="full"
)

print("\nFitting PCA on training data...")

# Fit PCA only on training data
X_train_pca = pca.fit_transform(X_train_scaled)

# Transform test data using the same fitted PCA
X_test_pca = pca.transform(X_test_scaled)

# Save fitted PCA for future predictions
joblib.dump(pca, "models/pca.pkl")

# Save transformed datasets
joblib.dump(
    (X_train_pca, X_test_pca, y_train, y_test),
    OUTPUT_FILE,
    compress=3
)

print("\n" + "=" * 50)
print("PCA COMPLETED")
print("=" * 50)

print("Original features:", X_train_scaled.shape[1])
print("Principal components:", X_train_pca.shape[1])

print(
    "Explained variance:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

print("\nTraining shape after PCA:", X_train_pca.shape)
print("Testing shape after PCA:", X_test_pca.shape)

print("\nSaved PCA model: models/pca.pkl")
print("Saved transformed data:", OUTPUT_FILE)
