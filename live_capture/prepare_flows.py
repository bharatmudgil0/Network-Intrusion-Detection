
import pandas as pd
import joblib
import os

# Input and output files
input_path = "live_capture/flows/captured_flows.csv"
output_path = "live_capture/flows/model_ready_flows.csv"

# Load CICFlowMeter output
df = pd.read_csv(input_path)

# Map model feature names to CICFlowMeter column names
feature_mapping = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "CWE Flag Count": "cwr_flag_count",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Header Length.1": "fwd_header_len",
    "Fwd Avg Bytes/Bulk": "fwd_byts_b_avg",
    "Fwd Avg Packets/Bulk": "fwd_pkts_b_avg",
    "Fwd Avg Bulk Rate": "fwd_blk_rate_avg",
    "Bwd Avg Bytes/Bulk": "bwd_byts_b_avg",
    "Bwd Avg Packets/Bulk": "bwd_pkts_b_avg",
    "Bwd Avg Bulk Rate": "bwd_blk_rate_avg",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min"
}

# Load the exact feature order used during model training
bundle = joblib.load("models/final_nids_model.pkl")
required_features = list(bundle["feature_names"])

# Check whether all source columns exist
missing_sources = [
    source for source in feature_mapping.values()
    if source not in df.columns
]

if missing_sources:
    print("ERROR: Required source columns are missing:")
    print(sorted(set(missing_sources)))
    raise SystemExit(1)

# Build model input in the original training feature order
model_df = pd.DataFrame(index=df.index)

for feature in required_features:
    if feature not in feature_mapping:
        print("ERROR: No mapping defined for:", feature)
        raise SystemExit(1)

    source_column = feature_mapping[feature]
    model_df[feature] = df[source_column]

# Convert features to numeric values
model_df = model_df.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with NaN
model_df = model_df.replace([float("inf"), float("-inf")], float("nan"))

# Report invalid values; do not silently invent replacements
invalid_rows = model_df.isna().any(axis=1)

print("\n========== MODEL FEATURE PREPARATION ==========")
print("Extracted flow records:", len(df))
print("Required model features:", len(required_features))
print("Prepared columns:", len(model_df.columns))
print("Rows with missing/invalid values:", int(invalid_rows.sum()))

if invalid_rows.any():
    print("\nInvalid values by feature:")
    print(model_df.isna().sum()[model_df.isna().sum() > 0])

# Save only if all features are present and valid
if invalid_rows.any():
    print("\nPreparation stopped. We must investigate invalid values first.")
    raise SystemExit(1)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
model_df.to_csv(output_path, index=False)

print("\nSUCCESS: Model-ready feature CSV created.")
print("Saved to:", output_path)
print("Final shape:", model_df.shape)
print("\nFeature order matches the saved model:", 
      list(model_df.columns) == required_features)
