# 🏦 Loan Approval Prediction App

This is a simple and beginner-friendly Machine Learning web app built with **Streamlit** that predicts whether a loan application is likely to be approved or rejected.

The app uses a **Random Forest Classifier** trained on a custom loan dataset and predicts approval based on user inputs such as income, CIBIL score, loan amount, and asset values.

---

[![Live Demo](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge)](https://loanapprv.streamlit.app/)

---

## 📂 Features

- Predict loan approval using a trained ML model.
- Interactive UI built with **Streamlit**.
- Input fields for education, employment status, financial assets, and more.
- Instant results with visual feedback.

---

## 🧠 Machine Learning Model

- Model: `RandomForestClassifier`
- Preprocessing:
  - Label Encoding for categorical columns.
  - Cleaned and encoded features for training.
- Training/Test split: `80/20`
- Accuracy & classification report printed in console during training.

---

## 📝 Input Features

- Education level
- Employment status (self-employed or not)
- CIBIL Score
- Income and Loan Details
- Value of residential, commercial, luxury, and bank assets
- Number of dependents

---

## 🛠️ Tech Stack

- **Python**
- **scikit-learn**
- **Streamlit**
- **pandas**
- **joblib**

---

## 🔧 Setup Instructions

1. **Clone the repository**  
   ```
   git clone https://github.com/rd9437/predicting_loan_approval.git
   cd predicting_loan_approval
   ```
