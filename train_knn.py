
import pandas as pd
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CREATE RESULTS DIRECTORY
# ==========================================

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==========================================
# 2. LOAD SMOTE DATASET
# ==========================================

print("Loading SMOTE dataset...")

data = joblib.load(
    "dataset/processed/splits/smote_data.pkl"
)

# Correct tuple order:
# data[0] = X_train
# data[1] = X_test
# data[2] = y_train
# data[3] = y_test

X_train, X_test, y_train, y_test = data

print("\nDataset loaded successfully!")

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing features:", X_test.shape)
print("Testing labels:", y_test.shape)


# ==========================================
# 3. INITIALIZE KNN MODEL
# ==========================================

print("\nInitializing KNN model...")

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    n_jobs=-1
)


# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining KNN...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 5. PREDICTION
# ==========================================

print("\nPredicting test records...")

y_pred = model.predict(X_test)


# ==========================================
# 6. MODEL EVALUATION
# ==========================================

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

weighted_f1 = f1_score(
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
print("KNN MODEL PERFORMANCE")
print("=" * 55)

print(f"Accuracy:     {accuracy * 100:.2f}%")
print(f"Precision:    {precision * 100:.2f}%")
print(f"Recall:       {recall * 100:.2f}%")
print(f"Weighted F1:  {weighted_f1 * 100:.2f}%")
print(f"Macro F1:     {macro_f1 * 100:.2f}%")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 8. SAVE CONFUSION MATRIX
# ==========================================

labels = sorted(pd.Series(y_test).unique())

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
    "results/knn_confusion_matrix.csv"
)


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

predictions = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

predictions.to_csv(
    "results/knn_predictions.csv",
    index=False
)


# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    "models/knn.pkl"
)

print("\nModel saved: models/knn.pkl")
print("Confusion matrix saved: results/knn_confusion_matrix.csv")
print("Predictions saved: results/knn_predictions.csv")

print("\nKNN training and evaluation completed successfully!")
