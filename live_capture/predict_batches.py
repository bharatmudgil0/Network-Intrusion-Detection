
import os
import glob
import time
import joblib
import pandas as pd
import numpy as np

from datetime import datetime

# =====================================================
# FOLDER CONFIGURATION
# =====================================================

FLOW_FOLDER = "live_capture/flows"
RESULT_FOLDER = "live_capture/results"
PROCESSED_FOLDER = "live_capture/predicted"

os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(
    RESULT_FOLDER,
    "batch_predictions.csv"
)

CHECK_INTERVAL = 5


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

print("[MODEL] Loading trained model...", flush=True)

bundle = joblib.load("models/final_nids_model.pkl")

scaler = bundle["scaler"]
pca = bundle["pca"]
model = bundle["model"]

required_features = list(bundle["feature_names"])

print("[MODEL] Random Forest model loaded.", flush=True)


# =====================================================
# PROCESS NEW FLOW FILES
# =====================================================

def predict_new_batches():

    flow_files = [
        file
        for file in glob.glob(
            os.path.join(FLOW_FOLDER, "live_*.csv")
        )
        if os.path.getsize(file) > 0
    ]

    for flow_file in sorted(flow_files):

        filename = os.path.basename(flow_file)

        marker = os.path.join(
            PROCESSED_FOLDER,
            filename + ".done"
        )

        if os.path.exists(marker):
            continue

        # Ensure the flow CSV is completely written
        if filename + ".done" in os.listdir(PROCESSED_FOLDER):
            continue

        print("\n[PREDICTION] Processing:", filename, flush=True)

        try:

            raw_df = pd.read_csv(flow_file)

            if raw_df.empty:
                print("[WARNING] Empty flow file. Skipping.")
                continue

            # Import feature mapping
            from prepare_flows import feature_mapping

            # Validate source columns
            missing = [
                feature_mapping[feature]
                for feature in required_features
                if feature not in feature_mapping
                or feature_mapping[feature] not in raw_df.columns
            ]

            if missing:
                print("[ERROR] Missing source columns:", missing)
                continue

            # Prepare features in model training order
            model_df = pd.DataFrame(index=raw_df.index)

            for feature in required_features:
                model_df[feature] = raw_df[
                    feature_mapping[feature]
                ]

            model_df = model_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            model_df = model_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            if model_df.isna().any().any():
                print("[ERROR] Invalid features found. Skipping batch.")
                continue

            # Apply preprocessing
            X_scaled = scaler.transform(model_df)
            X_pca = pca.transform(X_scaled)

            # Predict
            predictions = model.predict(X_pca)

            if hasattr(model, "predict_proba"):
                confidence = (
                    model.predict_proba(X_pca).max(axis=1) * 100
                )
            else:
                confidence = [None] * len(predictions)

            # Prepare output
            results = pd.DataFrame({

                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),

                "src_ip": raw_df["src_ip"].values,
                "dst_ip": raw_df["dst_ip"].values,
                "src_port": raw_df["src_port"].values,
                "dst_port": raw_df["dst_port"].values,

                "prediction": predictions,
                "confidence": confidence

            })

            # Append predictions
            results.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=not os.path.exists(OUTPUT_FILE),
                index=False
            )

            # Mark batch complete only after successful prediction
            with open(marker, "w") as f:
                f.write(datetime.now().isoformat())

            benign_count = sum(
                str(x).upper() == "BENIGN"
                for x in predictions
            )

            print(
                "[PREDICTION] Total flows:",
                len(results),
                flush=True
            )

            print(
                "[PREDICTION] Benign:",
                benign_count,
                flush=True
            )

            print(
                "[PREDICTION] Potential attacks:",
                len(results) - benign_count,
                flush=True
            )

        except Exception as error:

            print(
                "[ERROR] Prediction failed:",
                error,
                flush=True
            )


# =====================================================
# CONTINUOUS PREDICTION LOOP
# =====================================================

if __name__ == "__main__":

    print("=" * 50)
    print("ML-NIDS PREDICTION WORKER STARTED")
    print("Watching for new flow CSV files...")
    print("Press Ctrl+C to stop.")
    print("=" * 50)

    try:

        while True:

            predict_new_batches()

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:

        print("\n[PREDICTION] Worker stopped by user.")
