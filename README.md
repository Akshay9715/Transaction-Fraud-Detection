# 🚨 Fraud Detection System (Pre & Post Transaction)

This project implements a **Machine Learning powered Fraud Detection System** designed to identify fraudulent activities in financial transactions. It provides both **pre-transaction analysis** (risk estimation before execution) and **post-transaction analysis** (fraud detection after execution).  

The project leverages **data cleaning, feature engineering, and ML model building** to create reliable fraud detection pipelines. Interactive **Streamlit applications** allow real-time predictions with user inputs, making the system practical and easy to use.  

---

## 🌐 Live Demo
- 🔹 **Pre-Transaction Fraud Detection App**: [Try it here](https://pre-transaction-fraud-detection.streamlit.app/)  
- 🔹 **Post-Transaction Fraud Detection App**: [Try it here](https://post-transaction-fraud-detection.streamlit.app/)  

---

## 📌 Features
- **Pre-Transaction Module**
  - Predicts fraud risk **before the transaction occurs**.
  - Uses transaction metadata (amount, account details, type, etc.) to evaluate risk.
  - Helps prevent fraud proactively.

- **Post-Transaction Module**
  - Detects fraud **after the transaction** based on balance inconsistencies, transaction patterns, and engineered features.
  - Useful for auditing and real-time fraud alerts.

- **Feature Engineering**
  - Derived features like:
    - `isCashOutOrTransfer`
    - `step_mod_day`
    - `step_week`
    - `errBalanceOrig`
    - `errBalanceDest`
    - `log_amount`
    - `amt_over_oldOrig`
    - `orig_tx_count_sofar`
    - `dest_tx_count_sofar`

- **Interactive UI**
  - User-friendly **Streamlit apps** for entering transaction details.
  - Instant fraud prediction output.

---

## 🏗️ Tech Stack
- **Programming Language**: Python  
- **Frameworks & Libraries**:  
  - Data Processing: `pandas`, `numpy`  
  - ML: `scikit-learn`, `xgboost`, `joblib`  
  - Deployment: `Streamlit`  
- **Model Files**:  
  - `Pre_model.pkl`  
  - `Post_model.pkl`

---

## 🚀 Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
streamlit run pre_transaction_app.py
# OR
streamlit run post_transaction_app.py
```


### 📊 Dataset

The system was trained on financial transaction datasets, containing fields such as:

Transaction type (CASH_OUT, TRANSFER, etc.)

Amount, sender, and receiver details

Old and new account balances

Engineered features for anomaly detection

🔮 Future Improvements

Integration with real-time payment APIs for production use.

Explainable AI to show why a transaction was flagged.

Adding deep learning models (RNN/LSTM) for sequence-based fraud detection.

👨‍💻 Author
Developed by Akshay Choudhary

