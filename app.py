# app.py
import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ---------------- LOAD MODEL & PREPROCESSORS ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("sex_encoder.pkl", "rb") as f:
    sex_encoder = pickle.load(f)

with open("embarked_encoder.pkl", "rb") as f:
    embarked_encoder = pickle.load(f)

# ---------------- CUSTOM STYLING ----------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f8ff;
    }
    h1 {
        color: #1f4ed8;
        text-align: center;
    }
    .stButton>button {
        background-color: #4da6ff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.title("🚢 Titanic Survival Prediction")
st.caption("An educational ML project – not for real-life use")
st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("👤 Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)

with col2:
    fare = st.number_input("Fare", min_value=0.0, value=50.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Survival"):
    # Encode categorical
    sex_encoded = sex_encoder.transform([sex])[0]
    embarked_encoded = embarked_encoder.transform([embarked])[0]

    # Prepare and scale input
    input_data = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction] * 100

    # Display result
    if prediction == 1:
        st.success(f"🎉 Survived\nConfidence: {probability:.2f}%")
    else:
        st.error(f"❌ Did Not Survive\nConfidence: {probability:.2f}%")

st.markdown("---")
st.caption("Made with 💙 using Streamlit & Scikit-learn")
