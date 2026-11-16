
import os
import joblib
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

INPUT_FILE = "dataset/processed/splits/smote_data.pkl"
MODEL_FILE = "models/svm.pkl"
RESULT_DIR = "results"

os.makedirs("models", exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading SMOTE dataset...")

X_train, X_test, y_train, y_test = joblib.load(INPUT_FILE)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Create Linear SVM
model = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42
)

print("\nTraining Linear SVM...")

model.fit(X_train, y_train)

print("Training completed!")

print("\nPredicting test records...")

y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 55)
print("LINEAR SVM PERFORMANCE")
print("=" * 55)

print(f"Accuracy:     {accuracy * 100:.2f}%")
print(f"Precision:    {precision * 100:.2f}%")
print(f"Recall:       {recall * 100:.2f}%")
print(f"Weighted F1:  {weighted_f1 * 100:.2f}%")
print(f"Macro F1:     {macro_f1 * 100:.2f}%")

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

pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(
    os.path.join(RESULT_DIR, "svm_confusion_matrix.csv")
)

# Save predictions
pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
}).to_csv(
    os.path.join(RESULT_DIR, "svm_predictions.csv"),
    index=False
)

# Save trained model
joblib.dump(model, MODEL_FILE)

print("\nModel saved:", MODEL_FILE)
print("Results saved in:", RESULT_DIR)
