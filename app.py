import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

st.title("Employee Attrition Prediction")

# Load model & features
model = pickle.load(open("models/attrition_model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))

st.subheader("Enter Employee Details")

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

    st.success("Prediction Completed")
    st.write("**Attrition:**", "Yes" if prediction == 1 else "No")
    st.write(f"**Confidence:** {confidence:.2f}%")

st.subheader("Model Evaluation (Test Data)")

# Load dataset again ONLY for evaluation
df = pd.read_csv("data/HR-Employee-Attrition.csv")
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# ----- Confusion Matrix -----
cm = confusion_matrix(y, y_pred)

fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)

# ----- ROC Curve -----
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0, 1], [0, 1], linestyle="--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()
st.pyplot(fig2)
