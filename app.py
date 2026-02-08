import streamlit as st
import pickle
import pandas as pd
import os

st.title("Employee Attrition Prediction")

try:
    model = pickle.load(open("models/attrition_model.pkl", "rb"))
    features = pickle.load(open("models/features.pkl", "rb"))
except Exception as e:
    st.error("Model files could not be loaded")
    st.error(e)
    st.stop()

age = st.number_input("Age", 18, 60)
monthly_income = st.number_input("Monthly Income", 1000, 200000)
job_level = st.number_input("Job Level", 1, 5)
years_at_company = st.number_input("Years At Company", 0, 40)

if st.button("Predict"):
    input_data = dict.fromkeys(features, 0)

    if "Age" in input_data:
        input_data["Age"] = age
    if "MonthlyIncome" in input_data:
        input_data["MonthlyIncome"] = monthly_income
    if "JobLevel" in input_data:
        input_data["JobLevel"] = job_level
    if "YearsAtCompany" in input_data:
        input_data["YearsAtCompany"] = years_at_company

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    confidence = max(model.predict_proba(input_df)[0]) * 100

    st.success("Prediction Successful")
    st.write("Attrition:", "Yes" if prediction == 1 else "No")
    st.write(f"Confidence: {confidence:.2f}%")
