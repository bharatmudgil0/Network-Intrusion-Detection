
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Model prediction files
models = {
    "Decision Tree": "results/decision_tree_predictions.csv",
    "Random Forest": "results/random_forest_predictions.csv",
    "SVM": "results/svm_predictions.csv",
    "Logistic Regression": "results/logistic_regression_predictions.csv",
    "KNN": "results/knn_predictions.csv"
}

results = []

print("Comparing all trained models...\n")

for model_name, file_path in models.items():

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    y_true = df["Actual"]
    y_pred = df["Predicted"]

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "Weighted F1 (%)": round(weighted_f1 * 100, 2),
        "Macro F1 (%)": round(macro_f1 * 100, 2)
    })


# Create comparison table
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="Macro F1 (%)",
    ascending=False
)

print("=" * 85)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 85)

print(comparison.to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison saved: results/model_comparison.csv")

if not comparison.empty:
    print("\nBest model based on Macro F1:")
    print(comparison.iloc[0]["Model"])
