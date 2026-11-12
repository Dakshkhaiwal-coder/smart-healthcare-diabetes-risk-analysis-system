from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="Smart Healthcare Risk Analysis", page_icon="🩺", layout="wide")

st.title("🩺 Smart Healthcare Risk Analysis System")
st.subheader("Diabetes Risk Prediction Using Machine Learning")
st.write("Enter the required health measurements to view the machine-learning model's output.")

st.warning("Educational project only. This tool is not a medical device and cannot diagnose diabetes or guide treatment.")

if not Path("diabetes_model.pkl").exists():
    st.error("Model file not found. Run `python train_model.py` first to create diabetes_model.pkl.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load("diabetes_model.pkl")

model = get_model()

st.markdown("### Enter patient measurements")
with st.form("health_form"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose level", min_value=0, max_value=300, value=120)
        bp = st.number_input("Blood pressure", min_value=0, max_value=200, value=70)
        skin = st.number_input("Skin thickness", min_value=0, max_value=100, value=20)
    with c2:
        insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)
        pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
    submitted = st.form_submit_button("Analyse Risk", type="primary", use_container_width=True)

if submitted:
    values = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree, "Age": age
    }], columns=FEATURES)
    values[ZERO_AS_MISSING] = values[ZERO_AS_MISSING].replace(0, float("nan"))

    prediction = int(model.predict(values)[0])
    score = float(model.predict_proba(values)[0][1])

    st.markdown("---")
    st.markdown("### Analysis result")
    st.metric("Model-estimated positive-class score", f"{score * 100:.1f}%")
    if prediction == 1:
        st.error("The model classified this entry as Outcome 1 (positive class).")
    else:
        st.success("The model classified this entry as Outcome 0 (negative class).")

    st.caption("The score is a model output, not a clinically validated personal probability. A healthcare professional can advise on appropriate laboratory testing and interpretation.")

st.markdown("---")
st.caption("Minor Project | Daksh (2554670) and Deepanshu (2554675) | Python • Scikit-learn • Streamlit")
