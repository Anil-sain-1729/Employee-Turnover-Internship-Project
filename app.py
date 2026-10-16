from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "employee_turnover_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"
DATA_PATH = ROOT / "data" / "employee_data.csv"

st.set_page_config(page_title="Employee Turnover Predictor", page_icon="📊", layout="centered")
st.title("📊 Employee Turnover Prediction")
st.write(
    "Enter employee-related feature values to get a model estimate. "
    "This demo is for learning and must not be used as the sole basis for employment decisions."
)

if not MODEL_PATH.exists():
    st.warning("Model file not found. First run `python train_model.py` from the project folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
info = json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else {}
st.caption(f"Model used: {info.get('selected_model', 'trained classifier')}")

departments = ["sales", "technical", "support", "IT", "product_mng", "marketing",
               "RandD", "accounting", "hr", "management"]
salaries = ["low", "medium", "high"]

with st.form("employee_form"):
    st.subheader("Employee details")
    c1, c2 = st.columns(2)
    with c1:
        satisfaction = st.slider("Satisfaction level", 0.0, 1.0, 0.60, 0.01)
        evaluation = st.slider("Last evaluation", 0.0, 1.0, 0.70, 0.01)
        projects = st.number_input("Number of projects", 1, 10, 3)
        monthly_hours = st.number_input("Average monthly hours", 50, 400, 160)
    with c2:
        years = st.number_input("Years at company", 1, 40, 3)
        accident = st.selectbox("Work accident recorded?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        promotion = st.selectbox("Promotion in last 5 years?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        department = st.selectbox("Department", departments)
        salary = st.selectbox("Salary category", salaries)
    submitted = st.form_submit_button("Predict turnover risk", type="primary")

if submitted:
    sample = pd.DataFrame([{
        "satisfaction_level": satisfaction,
        "last_evaluation": evaluation,
        "number_project": int(projects),
        "average_montly_hours": int(monthly_hours),
        "time_spend_company": int(years),
        "Work_accident": int(accident),
        "promotion_last_5years": int(promotion),
        "department": department,
        "salary": salary
    }])
    probability = float(model.predict_proba(sample)[0, 1])
    prediction = int(probability >= 0.5)
    st.subheader("Prediction")
    st.metric("Estimated probability of leaving", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if prediction:
        st.error("The model estimates a higher likelihood of leaving (class 1).")
    else:
        st.success("The model estimates a lower likelihood of leaving (class 0).")
    st.caption("This is a statistical estimate, not a certainty. Consider context, data quality, and fairness.")

with st.expander("About the dataset"):
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        st.write(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
        st.dataframe(data.head(10), use_container_width=True)
    else:
        st.info("Dataset not found.")
