
import pandas as pd
import joblib
import os
from datetime import datetime

# Load trained model bundle
bundle = joblib.load("models/final_nids_model.pkl")

scaler = bundle["scaler"]
pca = bundle["pca"]
model = bundle["model"]

# Load prepared flow data
input_path = "live_capture/flows/model_ready_flows.csv"

df = pd.read_csv(input_path)

# Validate feature order
required_features = list(bundle["feature_names"])

if list(df.columns) != required_features:
    raise ValueError("Feature columns do not match model training order.")

# Apply the original preprocessing pipeline
X_scaled = scaler.transform(df)
X_pca = pca.transform(X_scaled)

# Predict
predictions = model.predict(X_pca)

# Calculate confidence where supported
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_pca)
    confidence = probabilities.max(axis=1) * 100
else:
    confidence = [None] * len(predictions)

# Create prediction results
results = pd.DataFrame({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "prediction": predictions,
    "confidence": confidence
})

# Add flow information from original CICFlowMeter output
raw_flows = pd.read_csv(
    "live_capture/flows/captured_flows.csv"
)

results["src_ip"] = raw_flows["src_ip"].values
results["dst_ip"] = raw_flows["dst_ip"].values

results["src_port"] = raw_flows["src_port"].values
results["dst_port"] = raw_flows["dst_port"].values

# Save predictions
os.makedirs("live_capture/results", exist_ok=True)

output_path = "live_capture/results/live_predictions.csv"

if os.path.exists(output_path):
    results.to_csv(output_path, mode="a", header=False, index=False)
else:
    results.to_csv(output_path, index=False)

print("\n========== LIVE FLOW PREDICTIONS ==========")
print(results.head(10).to_string(index=False))

print("\nTotal flows:", len(results))
print("Benign flows:", sum(str(x).upper() == "BENIGN" for x in predictions))
print("Potential attacks:", sum(str(x).upper() != "BENIGN" for x in predictions))
print("\nSaved predictions to:", output_path)
