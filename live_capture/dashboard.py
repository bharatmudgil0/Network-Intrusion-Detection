import os
import pandas as pd
import numpy as np
import streamlit as st

LIVE_FILE = "live_capture/live_predictions.csv"


@st.fragment(run_every="2s")
def update_live_dashboard():

    if not os.path.exists(LIVE_FILE):
        st.info("Waiting for live network predictions...")
        return

    df = pd.read_csv(LIVE_FILE)

    if df.empty:
        st.info("No network flows detected yet.")
        return

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp"])

    if df.empty:
        return

    df["Traffic Type"] = np.where(
        df["Predicted Attack"] == "BENIGN",
        "Benign",
        "Potential Attack"
    )

    total = len(df)
    benign = (df["Traffic Type"] == "Benign").sum()
    attacks = total - benign

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Flows", total)
    col2.metric("Benign Flows", int(benign))
    col3.metric("Potential Attacks", int(attacks))

    traffic = (
        df.set_index("Timestamp")
        .groupby("Traffic Type")
        .resample("10s")
        .size()
        .unstack(level=0, fill_value=0)
    )

    st.subheader("📈 Live Network Traffic")

    st.line_chart(
        traffic.tail(60),
        use_container_width=True
    )

    st.subheader("🔍 Recent Predictions")

    st.dataframe(
        df.sort_values(
            "Timestamp",
            ascending=False
        ).head(20),
        use_container_width=True
    )