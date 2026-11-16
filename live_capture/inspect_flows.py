import pandas as pd
import joblib

import os
from scapy.config import conf

conf.prog.tcpdump = os.path.abspath(
    "live_capture/tools/WinDump.exe"
)

# =====================================================
# LOAD GENERATED FLOW CSV
# =====================================================

file_path = "live_capture/flows/captured_flows.csv"

df = pd.read_csv(file_path)

print("\n========== FLOW DATASET INFORMATION ==========\n")

print("Total flow records:", len(df))

print("Total columns:", len(df.columns))

print("\nColumn names:\n")

for index, column in enumerate(df.columns):
    print(index + 1, column)


# =====================================================
# LOAD MODEL FEATURE NAMES
# =====================================================

bundle = joblib.load("models/final_nids_model.pkl")

required_features = list(bundle["feature_names"])

print("\n========== MODEL FEATURE VALIDATION ==========\n")

missing_features = [
    feature for feature in required_features
    if feature not in df.columns
]

extra_features = [
    feature for feature in df.columns
    if feature not in required_features
]

print("Required model features:", len(required_features))

print("Missing features:", len(missing_features))

print("Extra features:", len(extra_features))

if missing_features:

    print("\nMissing feature names:\n")

    for feature in missing_features:
        print(feature)

else:

    print("\nAll required model features are available.")

print("\n========== FLOW DATA PREVIEW ==========\n")

print(df.head())