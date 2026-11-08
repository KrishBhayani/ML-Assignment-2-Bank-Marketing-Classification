
# ============================================================
# Bank Marketing - Classification Model Evaluation App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 1. Application Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.title("🏦 Bank Marketing Classification")
st.write(
    "Compare five machine learning classification models "
    "using an uploaded test dataset."
)

# ------------------------------------------------------------
# 2. Load Saved Preprocessing Artifacts
# ------------------------------------------------------------

@st.cache_resource
def load_artifacts():

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    with open(
        MODEL_DIR / "feature_information.json",
        "r"
    ) as file:
        feature_info = json.load(file)

    return preprocessor, label_encoder, feature_info


# ------------------------------------------------------------
# 3. Load Selected Classification Model
# ------------------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}


@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


# ------------------------------------------------------------
# 4. Model Selection
# ------------------------------------------------------------

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

st.sidebar.info(
    "The application uses previously trained models. "
    "No model training is performed on uploaded test data."
)

# ------------------------------------------------------------
# 5. Upload Test Dataset
# ------------------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing input features and target column (y)",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # Load artifacts and model
        preprocessor, label_encoder, feature_info = load_artifacts()

        model_path = MODEL_DIR / MODEL_FILES[selected_model]

        model = load_model(str(model_path))

        # Read uploaded CSV
        test_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")

        st.dataframe(test_df.head())

        # ----------------------------------------------------
        # 6. Validate Uploaded Dataset
        # ----------------------------------------------------

        required_features = feature_info["input_features"]
        target_column = feature_info["target_column"]

        missing_features = [
            column for column in required_features
            if column not in test_df.columns
        ]

        if missing_features:
            st.error(
                f"Missing required features: {missing_features}"
            )
            st.stop()

        if target_column not in test_df.columns:
            st.error(
                "The uploaded CSV must contain the target column 'y'."
            )
            st.stop()

        if test_df.empty:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        # Separate input features and actual target
        X_uploaded = test_df[required_features].copy()

        y_actual_labels = test_df[target_column]

        # Validate target labels
        try:
            y_actual = label_encoder.transform(y_actual_labels)
        except ValueError:
            st.error(
                "Invalid target values. Expected values: "
                f"{list(label_encoder.classes_)}"
            )
            st.stop()

        # ----------------------------------------------------
        # 7. Apply Saved Preprocessing
        # ----------------------------------------------------

        X_processed = preprocessor.transform(X_uploaded)

        # ----------------------------------------------------
        # 8. Generate Predictions
        # ----------------------------------------------------

        y_pred = model.predict(X_processed)

        y_probability = model.predict_proba(X_processed)[:, 1]

        # ----------------------------------------------------
        # 9. Calculate Evaluation Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(y_actual, y_pred)

        auc = roc_auc_score(
            y_actual,
            y_probability
        )

        precision = precision_score(
            y_actual,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_actual,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_actual,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_actual,
            y_pred
        )

        # ----------------------------------------------------
        # 10. Display Selected Model
        # ----------------------------------------------------

        st.subheader("Evaluation Results")

        st.write(f"**Selected Model:** {selected_model}")

        st.write(
            f"**Total Test Records:** {len(test_df)}"
        )

        # ----------------------------------------------------
        # 11. Display Six Evaluation Metrics
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # ----------------------------------------------------
        # 12. Confusion Matrix
        # ----------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_actual,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"Confusion Matrix - {selected_model}")

        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # 13. Classification Report
        # ----------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_actual,
            y_pred,
            labels=[0, 1],
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.style.format("{:.4f}")
        )

        # ----------------------------------------------------
        # 14. Prediction Preview
        # ----------------------------------------------------

        st.subheader("Prediction Results")

        prediction_df = X_uploaded.copy()

        prediction_df["Actual"] = label_encoder.inverse_transform(
            y_actual
        )

        prediction_df["Predicted"] = label_encoder.inverse_transform(
            y_pred
        )

        prediction_df["Subscription Probability"] = y_probability

        st.dataframe(prediction_df.head(20))

        st.success(
            "Model evaluation completed successfully!"
        )

    except Exception as error:

        st.error(f"An error occurred: {error}")

else:

    st.info(
        "Please upload a test CSV file to evaluate the selected model."
    )

st.markdown("---")

st.caption(
    "Bank Marketing Classification Project | "
    "Machine Learning Assignment 2"
)
