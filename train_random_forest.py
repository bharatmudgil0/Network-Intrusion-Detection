
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

INPUT_FILE = "dataset/processed/splits/smote_data.pkl"
MODEL_FILE = "models/random_forest.pkl"
RESULT_DIR = "results"

os.makedirs("models", exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading SMOTE dataset...")

X_train, X_test, y_train, y_test = joblib.load(INPUT_FILE)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

# Train model
print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# Predict test records
print("\nPredicting test records...")

y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 55)
print("RANDOM FOREST PERFORMANCE")
print("=" * 55)

print(f"Accuracy:     {accuracy * 100:.2f}%")
print(f"Precision:    {precision * 100:.2f}%")
print(f"Recall:       {recall * 100:.2f}%")
print(f"Weighted F1:  {f1 * 100:.2f}%")
print(f"Macro F1:     {macro_f1 * 100:.2f}%")

# Classification report
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Save confusion matrix
labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "random_forest_confusion_matrix.csv"
    )
)

# Save model
joblib.dump(model, MODEL_FILE)

# Save predictions
pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
}).to_csv(
    os.path.join(
        RESULT_DIR,
        "random_forest_predictions.csv"
    ),
    index=False
)

print("\nModel saved:", MODEL_FILE)
print("Results saved in:", RESULT_DIR)
