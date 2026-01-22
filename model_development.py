# model_development.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("titanic.csv")

# ---------------- SELECT FEATURES ----------------
# We'll choose 5 input features
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
target = "Survived"

X = df[features]
y = df[target]

# ---------------- HANDLE MISSING VALUES ----------------
X["Age"].fillna(X["Age"].median(), inplace=True)
X["Fare"].fillna(X["Fare"].median(), inplace=True)
X["Embarked"].fillna("S", inplace=True)  # most common

# ---------------- ENCODE CATEGORICALS ----------------
sex_encoder = LabelEncoder()
X["Sex"] = sex_encoder.fit_transform(X["Sex"])

embarked_encoder = LabelEncoder()
X["Embarked"] = embarked_encoder.fit_transform(X["Embarked"])

# ---------------- FEATURE SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------------- SAVE MODEL & PREPROCESSORS ----------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("sex_encoder.pkl", "wb") as f:
    pickle.dump(sex_encoder, f)

with open("embarked_encoder.pkl", "wb") as f:
    pickle.dump(embarked_encoder, f)

print("✅ Model and preprocessors saved successfully with pickle.")
