import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("Pre_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Pre-Transaction Fraud Alerting System")

# Take inputs
step = st.number_input("Step (time step)", min_value=0, step=1)
txn_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
nameOrig = st.text_input("Origin Name (e.g., C12345)")
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f")
nameDest = st.text_input("Destination Name (e.g., M12345 or C12345)")
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, format="%.2f")

# Create dataframe
df = pd.DataFrame([{
    "step": step,
    "type": txn_type,
    "amount": amount,
    "nameOrig": nameOrig,
    "oldbalanceOrg": oldbalanceOrg,
    "nameDest": nameDest,
    "oldbalanceDest": oldbalanceDest
}])

# ---- Feature Engineering ----
pre_transaction_df = df.copy()

# dest_is_merchant
pre_transaction_df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype('int8')

# isCashOutOrTransfer
pre_transaction_df['isCashOutOrTransfer'] = ((df['type'] == 'CASH_OUT') | (df['type'] == 'TRANSFER')).astype('int8')

# step_mod_day & step_week
pre_transaction_df['step_mod_day'] = df['step'] % 24
pre_transaction_df['step_week'] = df['step'] // 24 // 7

# log amount
pre_transaction_df['log_amount'] = np.log1p(pre_transaction_df['amount'])

# ratios
pre_transaction_df['amt_over_oldOrig'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)
pre_transaction_df['amt_over_oldDest'] = df['amount'] / (df['oldbalanceDest'] + 1e-6)

# Drop unnecessary columns
drop_cols = ['nameOrig', 'nameDest', 'amount', 'type']
pre_transaction_df = pre_transaction_df.drop(columns=drop_cols, errors="ignore")

# Add isFraud dummy column (model input requirement)
pre_transaction_df['isFraud'] = 0  

# Rearrange columns to match model input
# final_features = ['step','oldbalanceOrg','oldbalanceDest','isFraud',
#                   'dest_is_merchant','isCashOutOrTransfer','step_mod_day',
#                   'step_week','log_amount','amt_over_oldOrig','amt_over_oldDest']
final_features = ['step','oldbalanceOrg','oldbalanceDest',
                  'dest_is_merchant','isCashOutOrTransfer',
                  'step_mod_day','step_week','log_amount',
                  'amt_over_oldOrig','amt_over_oldDest']

pre_transaction_df = pre_transaction_df[final_features]

st.subheader("📊 Processed Features")
st.write(pre_transaction_df)

# ---- Prediction ----
if st.button("Predict Fraud?"):
    prediction = model.predict(pre_transaction_df)
    proba = model.predict_proba(pre_transaction_df)[:,1] if hasattr(model, "predict_proba") else None

    if prediction[0] == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Prob: {proba[0]:.2f})" if proba is not None else "🚨 Fraudulent Transaction Detected!")
    else:
        st.success(f"✅ Legit Transaction (Prob: {proba[0]:.2f})" if proba is not None else "✅ Legit Transaction")
