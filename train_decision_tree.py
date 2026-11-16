
import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# File paths
INPUT_FILE = "dataset/processed/splits/smote_data.pkl"
MODEL_FILE = "models/decision_tree.pkl"
RESULT_DIR = "results"

os.makedirs("models", exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading SMOTE dataset...")

X_train, X_test, y_train, y_test = joblib.load(INPUT_FILE)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Create Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

# Train the model
print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

print("Training completed!")

# Predict test labels
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

print("\n" + "=" * 55)
print("DECISION TREE PERFORMANCE")
print("=" * 55)

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

# Detailed class-wise report
print("\nClassification Report:\n")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

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
    os.path.join(RESULT_DIR, "decision_tree_confusion_matrix.csv")
)

# Save model
joblib.dump(model, MODEL_FILE)

# Save predictions and actual labels
pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
}).to_csv(
    os.path.join(RESULT_DIR, "decision_tree_predictions.csv"),
    index=False
)

print("\nModel saved:", MODEL_FILE)
print("Results saved in:", RESULT_DIR)
