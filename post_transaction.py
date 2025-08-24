import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained fraud detection model
model = joblib.load("Post_model.pkl")

st.title("💳 Post-Transaction Fraud Detection System")

st.write("Enter the transaction details below to check if it is fraudulent or legitimate.")

# User inputs
step = st.number_input("Step (time step in hours)", min_value=1, value=10)
tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
nameOrig = st.text_input("Origin Account ID (nameOrig)", "C12345")
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=4000.0)
nameDest = st.text_input("Destination Account ID (nameDest)", "M67890")
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=10000.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=11000.0)

# ---- Feature Engineering (same as training pipeline) ----

# 1. dest_is_merchant: if nameDest starts with 'M' -> merchant
dest_is_merchant = 1 if nameDest.startswith("M") else 0

# 2. isCashOutOrTransfer: if type is CASH_OUT or TRANSFER
isCashOutOrTransfer = 1 if tx_type in ["CASH_OUT", "TRANSFER"] else 0

# 3. step_mod_day: modulo to simulate day-hour grouping
step_mod_day = step % 24

# 4. step_week: week number from step
step_week = step // (24 * 7)

# 5. errBalanceOrig: discrepancy in origin balance
errBalanceOrig = oldbalanceOrg - amount - newbalanceOrig

# 6. errBalaceDest: discrepancy in destination balance
errBalaceDest = newbalanceDest - oldbalanceDest - amount

# 7. log_amount: log transformation to reduce skew
log_amount = np.log1p(amount)

# 8. amt_over_oldOrig: ratio of transaction amount to origin balance
amt_over_oldOrig = amount / oldbalanceOrg if oldbalanceOrg > 0 else 0

# 9 & 10. tx counts so far (not available in single transaction context, so set as 1)
orig_tx_count_sofar = 1
dest_tx_count_sofar = 1

# ---- Create feature vector ----
features = pd.DataFrame([{
    "dest_is_merchant": dest_is_merchant,
    "isCashOutOrTransfer": isCashOutOrTransfer,
    "step_mod_day": step_mod_day,
    "step_week": step_week,
    "errBalanceOrig": errBalanceOrig,
    "errBalaceDest": errBalaceDest,
    "log_amount": log_amount,
    "amt_over_oldOrig": amt_over_oldOrig,
    "orig_tx_count_sofar": orig_tx_count_sofar,
    "dest_tx_count_sofar": dest_tx_count_sofar
}])

# ---- Prediction ----
if st.button("Predict Fraud"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Fraud Probability: {proba:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction. (Fraud Probability: {proba:.2%})")
