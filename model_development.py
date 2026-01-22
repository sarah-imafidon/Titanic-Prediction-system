import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("titanic.csv")

# 2. Select required columns
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
target = "Survived"

df = df[features + [target]]

# 3. Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# 4. Encode categorical variables
sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

df["Sex"] = sex_encoder.fit_transform(df["Sex"])
df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

# 5. Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 9. Save model & preprocessors
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(sex_encoder, "sex_encoder.joblib")
joblib.dump(embarked_encoder, "embarked_encoder.joblib")

# 10. Reload test (proof)
loaded_model = joblib.load("model.joblib")
test_prediction = loaded_model.predict(X_test[:1])
print("Test reload prediction:", test_prediction)
