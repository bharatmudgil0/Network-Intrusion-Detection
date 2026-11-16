# Network-Intrusion-Detection
Machine Learning-Based Network Intrusion Detection System (ML-NIDS) using the CICIDS2017 dataset to classify network traffic and detect potential cyberattacks through multiple ML algorithms, with real-time packet capture, flow analysis, and a Streamlit dashboard.

# 🛡️ ML-Based Network Intrusion Detection System (ML-NIDS)

ML-NIDS is a machine learning-based Network Intrusion Detection System designed to analyze network traffic, classify benign and malicious network flows, and identify potential cyberattacks.

The project uses the **CICIDS2017 dataset** to train and evaluate multiple machine learning classification algorithms. It also incorporates live network packet capture, network-flow extraction, automated prediction, and an interactive Streamlit dashboard for monitoring network traffic.

## 🚀 Key Features

- **Multiple Machine Learning Models:** Random Forest, Decision Tree, K-Nearest Neighbors (KNN), Logistic Regression, and Support Vector Machine (SVM).
- **Network Traffic Classification:** Classifies network flows into BENIGN and multiple attack categories.
- **Data Preprocessing:** Handles missing values, duplicate records, feature scaling, and dimensionality reduction using PCA.
- **SMOTE:** Addresses class imbalance during model training.
- **Live Network Monitoring:** Captures network packets and extracts network flows for prediction.
- **Automated Intrusion Detection:** Applies trained ML models to extracted network-flow features.
- **Interactive Dashboard:** Streamlit dashboard displaying traffic statistics, prediction distribution, potential attack alerts, and recent predictions.
- **Model Evaluation:** Accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

## 🧰 Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- Streamlit
- Matplotlib
- Scapy
- CICFlowMeter
- Joblib
- CICIDS2017 Dataset

## 📊 Dataset

The project uses the CICIDS2017 dataset, which contains benign network traffic and various attack categories, including DDoS, DoS, PortScan, Bot, and Web Attacks.

## 🎯 Project Objective

The primary objective is to develop a machine learning-based approach for network intrusion detection that can analyze network flows, recognize known attack patterns, and provide an interactive environment for monitoring potential network threats.

**Note:** Model predictions are potential security alerts and require further investigation. Live detection performance and feature compatibility depend on the network-flow extraction pipeline.
