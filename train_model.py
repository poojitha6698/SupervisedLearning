import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/HR-Employee-Attrition.csv")

# Encode target
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Encode categorical features
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Save feature names
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model and features
pickle.dump(model, open("models/attrition_model.pkl", "wb"))
pickle.dump(feature_names, open("models/features.pkl", "wb"))
