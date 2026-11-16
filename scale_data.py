
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Load training and testing data
INPUT_FILE = "dataset/processed/splits/train_test_data.pkl"

OUTPUT_DIR = "dataset/processed/splits"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scaled_data.pkl")

print("Loading training and testing data...")

X_train, X_test, y_train, y_test = joblib.load(INPUT_FILE)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Create scaler
scaler = StandardScaler()

print("\nFitting StandardScaler on training data...")

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to testing data
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler for future predictions
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Save scaled datasets
joblib.dump(
    (X_train_scaled, X_test_scaled, y_train, y_test),
    OUTPUT_FILE,
    compress=3
)

print("\n" + "=" * 50)
print("FEATURE SCALING COMPLETED")
print("=" * 50)

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)

print("\nFirst five scaled training records:")
print(X_train_scaled[:5, :5])

print("\nSaved scaler: models/scaler.pkl")
print("Saved data:", OUTPUT_FILE)
