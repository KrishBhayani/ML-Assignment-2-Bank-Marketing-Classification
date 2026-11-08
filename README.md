# Bank Marketing Classification Using Machine Learning

## 1. Project Overview

This project implements five machine learning classification algorithms to predict whether a bank customer will subscribe to a term deposit based on customer information and previous marketing campaign details.

The objective is to compare the performance of different classification algorithms using multiple evaluation metrics and identify the best-performing model.

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

Dataset details:

- Total Records: 45,211
- Total Columns: 17
- Input Features: 16 originally
- Target Variable: `y`
- Classification Type: Binary Classification
- Target Classes: Yes and No

The target variable indicates whether a customer subscribed to a term deposit.

The `duration` feature was excluded from model training because it is only known after the marketing call ends.

## 3. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Data Preprocessing

The following preprocessing techniques were applied:

- Removed the `duration` feature to avoid data leakage.
- Separated input features and target variable.
- Converted target labels into numerical values.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features.
- Split the dataset into 80% training and 20% testing.
- Used stratified sampling to maintain class distribution.

Final processed feature count: 50.

## 5. Model Evaluation and Comparison

All five classification models were evaluated using the same test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8308 | 0.6094 | 0.2948 | 0.3204 | 0.3071 | 0.2111 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| SVM | 0.8933 | 0.7183 | 0.6555 | 0.1853 | 0.2889 | 0.3098 |

## 6. Model Observations

### Logistic Regression
- Achieved the highest AUC score of 0.7717.
- Achieved the highest precision of 0.6632.
- Accuracy was 0.8933.
- Recall was relatively low at 0.1786.

### Decision Tree
- Achieved an accuracy of 0.8308.
- Recall was 0.3204.
- Had the lowest AUC score of 0.6094.

### K-Nearest Neighbors
- Achieved an accuracy of 0.8874.
- Precision was 0.5529.
- Recall was 0.1975.

### Gaussian Naive Bayes
- Achieved the highest recall of 0.4641.
- Achieved the highest F1 Score of 0.4123.
- Achieved the highest MCC of 0.3271.
- Identified the highest number of actual subscribers correctly.

### Support Vector Machine
- Achieved an accuracy of 0.8933.
- Precision was 0.6555.
- Required approximately 212.69 seconds for training.

## 7. Overall Best Model

Gaussian Naive Bayes was selected as the overall best-performing model because it achieved the highest Recall, F1 Score, and Matthews Correlation Coefficient.

Since the dataset is imbalanced, F1 Score, Recall, and MCC provide useful information beyond accuracy.

Logistic Regression performed better in terms of AUC and Precision.

## 8. Streamlit Application

The project includes an interactive Streamlit application that provides:

- CSV test dataset upload.
- Selection of five trained classification models.
- Evaluation using six performance metrics.
- Confusion matrix visualization.
- Classification report.
- Prediction results.

The application loads previously saved models and does not retrain models on uploaded test data.

## 9. Project Structure

```text
Assignment 2/
│
├── app.py
├── ML_Assignment_2.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
│
├── dataset/
│   └── bank-full.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── gaussian_naive_bayes.pkl
│   ├── svm.pkl
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   └── feature_information.json
│
└── outputs/
    └── model_evaluation_results.csv
```

## 10. Installation and Execution

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 11. Project Links

**GitHub Repository:** [GitHub Repository URL](https://github.com/KrishBhayani/ML-Assignment-2-Bank-Marketing-Classification)

**Live Streamlit Application:** [Live Streamlit Application URL](https://ml-assignment-2-bank-marketing-classification-2mnuv8dcpptxrn7m.streamlit.app/)

## 12. Conclusion

This project demonstrates the implementation and comparison of five machine learning classification algorithms on the Bank Marketing dataset. The results show that model performance varies across evaluation metrics, particularly because of class imbalance. Gaussian Naive Bayes achieved the best balanced performance based on F1 Score, Recall, and MCC among the evaluated models.