import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

print("Loading Random Forest model...")

model = joblib.load("models/random_forest.pkl")

# Get importance of the 21 PCA components
importance = model.feature_importances_

feature_names = [
    f"PC{i+1}" for i in range(len(importance))
]

importance_df = pd.DataFrame({
    "Principal Component": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nPrincipal Component Importance:")
print(importance_df.to_string(index=False))

# Save results
importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)

# Plot top 15 components
top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Principal Component"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Principal Components")
plt.title("Random Forest - PCA Component Importance")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("results/feature_importance.csv")
print("results/feature_importance.png")