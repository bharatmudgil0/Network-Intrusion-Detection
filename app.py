
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ML-NIDS | Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Record when the current Streamlit server session starts

if "live_session_start" not in st.session_state:
    st.session_state.live_session_start = pd.Timestamp.now()


# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "models/final_nids_model.pkl"
    )


MODEL_PATHS = {
    "Random Forest": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "SVM": "models/svm.pkl"
}


@st.cache_resource
def load_algorithm_model(model_path):

    return joblib.load(model_path)


try:

    bundle = load_model()

    scaler = bundle["scaler"]
    pca = bundle["pca"]
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])

except Exception as e:

    st.error(f"Unable to load trained model: {e}")
    st.stop()


# =====================================================
# 3. SIDEBAR
# =====================================================

st.sidebar.title("🛡️ ML-NIDS")

st.sidebar.write(
    "Machine Learning-Based Network "
    "Intrusion Detection System"
)

st.sidebar.divider()

st.sidebar.subheader("⚙️ Select Prediction Algorithm")

available_algorithms = {
    name: path
    for name, path in MODEL_PATHS.items()
    if os.path.exists(path)
}

# Use the model from the final bundle if the
# separate Random Forest file is unavailable.

if "Random Forest" not in available_algorithms:

    available_algorithms["Random Forest"] = None


selected_algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    options=list(available_algorithms.keys()),
    index=list(available_algorithms.keys()).index(
        "Random Forest"
    )
)


if available_algorithms[selected_algorithm] is None:

    selected_model = model

else:

    try:

        selected_model = load_algorithm_model(
            available_algorithms[selected_algorithm]
        )

    except Exception as e:

        st.sidebar.error(
            f"Unable to load {selected_algorithm}: {e}"
        )

        st.stop()


st.sidebar.success(
    f"Selected Model: {selected_algorithm}"
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write(f"**Algorithm:** {selected_algorithm}")
st.sidebar.write("**Dataset:** CICIDS2017")
st.sidebar.write("**Input Features:** 78")
st.sidebar.write("**PCA Components:** 21")
st.sidebar.write("**Classification:** Multiclass")
st.sidebar.write("**Attack Categories:** 14 + BENIGN")

st.sidebar.divider()

st.sidebar.subheader("Random Forest Reference Performance")

st.sidebar.metric("Accuracy", "99.38%")
st.sidebar.metric("Weighted F1", "99.40%")
st.sidebar.metric("Macro F1", "88.85%")

st.sidebar.caption(
    "These reference metrics belong to the "
    "Random Forest experiment, not every selected algorithm."
)

st.sidebar.divider()

st.sidebar.caption(
    "Academic research implementation based on CICIDS2017."
)


# =====================================================
# 4. MAIN HEADER
# =====================================================

st.title(
    "🛡️ Network Intrusion Detection System"
)

st.write(
    """
    This system uses machine learning classifiers
    to analyze network-flow records and identify
    benign traffic and potential cyberattacks.
    """
)

st.info(
    "Upload a CSV containing the 78 CICIDS2017 network-flow "
    "features. The Label column is optional and is not used "
    "to generate predictions."
)


# =====================================================
# 4.1 DETECTION MODE SELECTION
# =====================================================

if "detection_mode" not in st.session_state:

    st.session_state.detection_mode = "LIVE DETECTION"


col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🟢 LIVE DETECTION",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.detection_mode == "LIVE DETECTION"
            else "secondary"
        )
    ):

        st.session_state.detection_mode = "LIVE DETECTION"
        st.rerun()


with col2:

    if st.button(
        "📂 UPLOAD FILE",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.detection_mode == "UPLOAD FILE"
            else "secondary"
        )
    ):

        st.session_state.detection_mode = "UPLOAD FILE"
        st.rerun()


st.divider()


# =====================================================
# 4.2 LIVE DETECTION DASHBOARD
# =====================================================

if st.session_state.detection_mode == "LIVE DETECTION":

    @st.fragment(run_every="2s")
    def live_detection_dashboard():

        st.header("🟢 Live Network Intrusion Detection")

        st.caption(
            "Displaying network-flow predictions collected "
            "from packet captures."
        )

        prediction_file = (
            "live_capture/results/batch_predictions.csv"
        )

        if not os.path.exists(prediction_file):

            st.warning(
                "Prediction file not found. Start the packet "
                "capture and prediction pipeline first."
            )

            return

        try:

            live_df = pd.read_csv(prediction_file)
            
            # Convert timestamps

            live_df["timestamp"] = pd.to_datetime(
                live_df["timestamp"],
                errors="coerce"
            )

            # Remove predictions from previous server sessions

            live_df = live_df[
                live_df["timestamp"] >= st.session_state.live_session_start
            ].copy()

            if live_df.empty:

                st.info(
                    "No network-flow predictions are available yet."
                )

                return

            required_columns = [
                "timestamp",
                "prediction"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in live_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing columns in prediction file: "
                    f"{missing_columns}"
                )

                return

            live_df["timestamp"] = pd.to_datetime(
                live_df["timestamp"],
                errors="coerce"
            )

            live_df = live_df.dropna(
                subset=["timestamp"]
            )

            if live_df.empty:

                st.info(
                    "No valid prediction timestamps are available."
                )

                return

            total_flows = len(live_df)

            benign_flows = (
                live_df["prediction"]
                .astype(str)
                .str.upper()
                .eq("BENIGN")
                .sum()
            )

            attack_flows = (
                total_flows - benign_flows
            )

            # Summary metrics

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Network Flows",
                f"{total_flows:,}"
            )

            col2.metric(
                "Benign Flows",
                f"{benign_flows:,}"
            )

            col3.metric(
                "Potential Attacks",
                f"{attack_flows:,}"
            )

            st.divider()

            # Network traffic graph

            st.subheader(
                "📈 Network Traffic Monitoring"
            )

            chart_data = (
                live_df
                .set_index("timestamp")
                .resample("10s")
                .size()
                .rename("Network Flows")
            )

            st.line_chart(chart_data)

            # Prediction distribution

            st.subheader(
                "📊 Prediction Distribution"
            )

            # Prediction distribution

            st.subheader("📊 Prediction Distribution")

            distribution = (
                live_df["prediction"]
                .value_counts()
                .sort_values(ascending=True)
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            bars = ax.barh(
                distribution.index.astype(str),
                distribution.values
            )

            ax.set_title("Network Traffic Prediction Distribution")
            ax.set_xlabel("Number of Network Flows")
            ax.set_ylabel("Prediction Category")

            # Display count at the end of each bar

            for bar in bars:

                width = bar.get_width()

                ax.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f" {int(width):,}",
                    va="center"
                )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            # Recent predictions

            st.subheader(
                "🔍 Recent Network Flow Predictions"
            )

            st.dataframe(
                live_df.sort_values(
                    "timestamp",
                    ascending=False
                ).head(20),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Non-BENIGN predictions are potential alerts, "
                "not confirmed security incidents."
            )

        except Exception as e:

            st.error(
                f"Unable to read live predictions: {e}"
            )


    live_detection_dashboard()

    # Do not display the upload interface in Live Detection mode.

    st.stop()


# =====================================================
# 5. MODEL PERFORMANCE OVERVIEW
# =====================================================

st.header("📊 Model Performance Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Selected Algorithm",
    selected_algorithm
)

col2.metric(
    "Input Features",
    len(feature_names)
)

col3.metric(
    "PCA Components",
    getattr(pca, "n_components_", "N/A")
)

col4.metric(
    "Classification",
    "Multiclass"
)

st.caption(
    "Actual evaluation metrics for the selected algorithm "
    "are calculated using the held-out test dataset below."
)

st.divider()


# =====================================================
# 6. UPLOAD NETWORK TRAFFIC DATASET
# =====================================================

st.header("📂 Upload Network Traffic Dataset")

uploaded_file = st.file_uploader(
    "Choose a network traffic CSV file",
    type=["csv"],
    key="traffic_upload"
)


if uploaded_file is not None:

    try:

        # Read uploaded CSV

        df = pd.read_csv(uploaded_file)

        # Clean column names

        df.columns = df.columns.str.strip()

        # Remove optional target column

        if "Label" in df.columns:

            df = df.drop(columns=["Label"])

        st.success(
            "Dataset uploaded successfully!"
        )

        # Dataset information

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Uploaded Records",
            f"{len(df):,}"
        )

        col2.metric(
            "Uploaded Features",
            len(df.columns)
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # =====================================================
        # 7. FEATURE VALIDATION
        # =====================================================

        missing_features = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing_features:

            st.error(
                f"Missing {len(missing_features)} required features."
            )

            st.write("Missing features:")
            st.write(missing_features)

            st.stop()

        # =====================================================
        # 8. PREPARE INPUT DATA
        # =====================================================

        X = df[feature_names].copy()

        # Convert features to numeric

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Replace infinite values

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Identify valid rows

        valid_mask = X.notna().all(axis=1)

        valid_data = X.loc[valid_mask]

        invalid_count = int(
            (~valid_mask).sum()
        )

        if invalid_count > 0:

            st.warning(
                f"{invalid_count} records were skipped because "
                "of missing, infinite or non-numeric values."
            )

        if len(valid_data) == 0:

            st.error(
                "No valid network-flow records were found."
            )

            st.stop()

        # =====================================================
        # 9. DETECT NETWORK INTRUSIONS
        # =====================================================

        if st.button(
            "🚀 Detect Network Intrusions",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                f"Analyzing network traffic using {selected_algorithm}..."
            ):

                # StandardScaler

                X_scaled = scaler.transform(
                    valid_data
                )

                # PCA

                X_pca = pca.transform(
                    X_scaled
                )

                # Predictions using selected algorithm

                predictions = selected_model.predict(
                    X_pca
                )

                # Prediction probabilities.
                # LinearSVC does not support predict_proba().

                if hasattr(selected_model, "predict_proba"):

                    probabilities = selected_model.predict_proba(
                        X_pca
                    )

                    confidence = (
                        np.max(probabilities, axis=1) * 100
                    )

                else:

                    confidence = np.full(
                        len(predictions),
                        np.nan
                    )

                # Create results dataframe

                results = pd.DataFrame(
                    index=valid_data.index
                )

                results["Predicted Attack"] = predictions

                results["Predicted Probability (%)"] = (
                    confidence.round(2)
                )

                results["Status"] = np.where(
                    results["Predicted Attack"] == "BENIGN",
                    "Benign",
                    "Potential Attack"
                )

                results = results.sort_index()

                # Save results in session state

                st.session_state["results"] = results

                st.session_state["invalid_count"] = (
                    invalid_count
                )

                st.session_state["uploaded_filename"] = (
                    uploaded_file.name
                )

                st.session_state["selected_algorithm"] = (
                    selected_algorithm
                )

                st.success(
                    "Network traffic analysis completed successfully!"
                )

    except Exception as e:

        st.error(
            f"Error processing uploaded dataset: {e}"
        )


# =====================================================
# 10. DISPLAY PREDICTION RESULTS
# =====================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    total_records = len(results)

    benign_count = int(
        (
            results["Predicted Attack"] == "BENIGN"
        ).sum()
    )

    attack_count = (
        total_records - benign_count
    )

    invalid_count = st.session_state.get(
        "invalid_count",
        0
    )

    st.divider()

    st.header(
        "🔍 Intrusion Detection Results"
    )

    st.caption(
        f"Analyzed file: "
        f"{st.session_state.get('uploaded_filename', '')}"
        f" | Algorithm: "
        f"{st.session_state.get('selected_algorithm', 'Unknown')}"
    )

    # =====================================================
    # 11. SUMMARY METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Processed Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Benign Traffic",
        f"{benign_count:,}"
    )

    col3.metric(
        "Potential Attacks",
        f"{attack_count:,}"
    )

    col4.metric(
        "Skipped Records",
        f"{invalid_count:,}"
    )

    # =====================================================
    # 12. PREDICTION DETAILS
    # =====================================================

    st.subheader(
        "📋 Prediction Details"
    )

    st.dataframe(
        results,
        use_container_width=True
    )

    # =====================================================
    # 13. ATTACK-WISE SECURITY ANALYSIS
    # =====================================================

    st.divider()

    st.header(
        "🔍 Attack-wise Security Analysis"
    )

    attack_summary = (
        results.groupby("Predicted Attack")
        .agg(
            Records=("Predicted Attack", "count"),
            Average_Probability=(
                "Predicted Probability (%)",
                "mean"
            )
        )
        .reset_index()
    )

    attack_summary["Percentage (%)"] = (
        attack_summary["Records"]
        / total_records * 100
    )

    attack_summary["Average_Probability"] = (
        attack_summary["Average_Probability"].round(2)
    )

    attack_summary["Percentage (%)"] = (
        attack_summary["Percentage (%)"].round(2)
    )

    attack_summary = attack_summary.rename(
        columns={
            "Predicted Attack": "Attack Category",
            "Average_Probability": "Average Probability (%)"
        }
    )

    st.dataframe(
        attack_summary.sort_values(
            by="Records",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # 14. TRAFFIC CLASSIFICATION DISTRIBUTION
    # =====================================================

    st.subheader(
        "📊 Traffic Classification Distribution"
    )

    distribution = (
        results["Predicted Attack"].value_counts()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    distribution.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_title(
        "Traffic Classification Distribution"
    )

    ax.set_xlabel("Number of Records")
    ax.set_ylabel("Attack Category")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # =====================================================
    # 15. BENIGN VS POTENTIAL ATTACK
    # =====================================================

    st.subheader(
        "🛡️ Benign Traffic vs Potential Attacks"
    )

    chart_data = pd.Series(
        {
            "Benign": benign_count,
            "Potential Attack": attack_count
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    chart_data.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Benign Traffic vs Potential Attacks"
    )

    ax.set_ylabel("Number of Records")
    ax.set_xlabel("Traffic Type")

    plt.xticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # =====================================================
    # 16. PREDICTION PROBABILITY ANALYSIS
    # =====================================================

    st.subheader(
        "📈 Prediction Probability Analysis"
    )

    probability_values = (
        results["Predicted Probability (%)"].dropna()
    )

    if not probability_values.empty:

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(
            probability_values,
            bins=20
        )

        ax.set_title(
            "Distribution of Prediction Probabilities"
        )

        ax.set_xlabel("Predicted Probability (%)")
        ax.set_ylabel("Number of Records")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "Prediction probability chart is unavailable "
            "for this algorithm because it does not provide "
            "probability estimates."
        )

    st.caption(
        "These probabilities are model estimates. They are not "
        "guaranteed measures of correctness or calibrated risk scores."
    )


# =====================================================
# 17. MODEL EVALUATION USING SELECTED ALGORITHM
# =====================================================

st.divider()

st.header(
    "📊 Model Evaluation Analysis"
)

st.write(
    f"""
    The following evaluation uses the held-out test dataset
    to evaluate the selected {selected_algorithm} algorithm.

    The same test records are used for every algorithm
    to ensure a fair comparison.
    """
)

test_data_path = (
    "dataset/processed/splits/smote_data.pkl"
)

if os.path.exists(test_data_path):

    try:

        # Load the saved test dataset

        test_data = joblib.load(test_data_path)

        X_train, X_test, y_train, y_test = test_data

        # Predict using the selected algorithm

        y_pred = selected_model.predict(X_test)

        y_true = y_test

        labels = sorted(
            pd.unique(
                pd.concat([
                    pd.Series(y_true),
                    pd.Series(y_pred)
                ])
            )
        )

        # =====================================================
        # 18. EVALUATION METRICS
        # =====================================================

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        weighted_precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        weighted_recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        st.subheader(
            "Overall Evaluation Metrics"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{weighted_precision * 100:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{weighted_recall * 100:.2f}%"
        )

        col4.metric(
            "Weighted F1",
            f"{weighted_f1 * 100:.2f}%"
        )

        col5.metric(
            "Macro F1",
            f"{macro_f1 * 100:.2f}%"
        )

        # =====================================================
        # 19. CLASSIFICATION REPORT
        # =====================================================

        st.subheader(
            "📋 Class-wise Classification Report"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        # Keep only individual class rows

        class_report = report_df.loc[
            labels,
            ["precision", "recall", "f1-score", "support"]
        ].copy()

        class_report = class_report.rename(
            columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1 Score",
                "support": "Support"
            }
        )

        st.dataframe(
            class_report.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}",
                "Support": "{:.0f}"
            }),
            use_container_width=True
        )

        # =====================================================
        # 20. CONFUSION MATRIX
        # =====================================================

        st.subheader(
            "🔢 Confusion Matrix"
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        fig, ax = plt.subplots(
            figsize=(14, 11)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=90,
            colorbar=True,
            values_format="d"
        )

        ax.set_title(
            f"{selected_algorithm} - Confusion Matrix"
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # =====================================================
        # 21. NORMALIZED CONFUSION MATRIX
        # =====================================================

        st.subheader(
            "📊 Normalized Confusion Matrix"
        )

        fig, ax = plt.subplots(
            figsize=(14, 11)
        )

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=labels,
            normalize="true",
            display_labels=labels,
            cmap="Blues",
            xticks_rotation=90,
            values_format=".2f",
            ax=ax
        )

        ax.set_title(
            f"{selected_algorithm} - Normalized Confusion Matrix"
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # =====================================================
        # 22. FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS
        # =====================================================

        st.subheader(
            "⚠️ False Positive and False Negative Analysis"
        )

        # Calculate one-vs-rest counts for each class

        diagonal = np.diag(cm)

        false_positive = (
            cm.sum(axis=0) - diagonal
        )

        false_negative = (
            cm.sum(axis=1) - diagonal
        )

        true_positive = diagonal

        false_analysis = pd.DataFrame({
            "Attack Category": labels,
            "True Positive": true_positive,
            "False Positive": false_positive,
            "False Negative": false_negative
        })

        st.dataframe(
            false_analysis,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        false_analysis.set_index(
            "Attack Category"
        )[
            ["False Positive", "False Negative"]
        ].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "False Positives vs False Negatives by Class"
        )

        ax.set_xlabel("Attack Category")
        ax.set_ylabel("Number of Records")

        plt.xticks(rotation=75)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # =====================================================
        # 23. DOWNLOAD EVALUATION REPORT
        # =====================================================

        st.subheader(
            "📥 Download Model Evaluation Reports"
        )

        report_csv = class_report.to_csv().encode(
            "utf-8"
        )

        st.download_button(
            label="⬇️ Download Classification Report",
            data=report_csv,
            file_name="nids_classification_report.csv",
            mime="text/csv"
        )

        confusion_df = pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )

        confusion_csv = confusion_df.to_csv().encode(
            "utf-8"
        )

        st.download_button(
            label="⬇️ Download Confusion Matrix",
            data=confusion_csv,
            file_name="nids_confusion_matrix.csv",
            mime="text/csv"
        )

        false_analysis_csv = false_analysis.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download FP/FN Analysis",
            data=false_analysis_csv,
            file_name="nids_false_positive_negative.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Unable to evaluate the selected model: {e}"
        )

else:

    st.error(
        "Test dataset not found. Check the path used "
        "in apply_smote.py."
    )


# =====================================================
# 24. FOOTER
# =====================================================

st.divider()

st.caption(
    "ML-Based Network Intrusion Detection System | "
    "CICIDS2017 Research Project"
)

st.caption(
    "This application performs network-flow classification. "
    "Model predictions are not confirmed security incidents. "
    "Investigate alerts before taking security action."
)
