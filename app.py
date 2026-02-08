import streamlit as st
import pickle
import pandas as pd

# Load model and feature names
model = pickle.load(open("models/attrition_model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))

st.title("Employee Attrition Prediction")

st.write("Enter employee details:")

# Inputs based on COMMON Kaggle columns
age = st.number_input("Age", 18, 60)
monthly_income = st.number_input("Monthly Income", 1000, 200000)
job_level = st.number_input("Job Level", 1, 5)
years_at_company = st.number_input("Years At Company", 0, 40)

if st.button("Predict"):

    # Create full feature dictionary with zeros
    input_data = dict.fromkeys(features, 0)

    # Fill known features (names MUST match dataset columns)
    input_data["Age"] = age
    input_data["MonthlyIncome"] = monthly_income
    input_data["JobLevel"] = job_level
    input_data["YearsAtCompany"] = years_at_company

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    confidence = max(model.predict_proba(input_df)[0]) * 100

    st.subheader("Prediction Result")
    st.write("Attrition:", "Yes" if prediction == 1 else "No")
    st.write(f"Confidence: {confidence:.2f}%")
