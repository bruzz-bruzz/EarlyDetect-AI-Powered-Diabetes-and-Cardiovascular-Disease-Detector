import streamlit as st
import joblib
import pandas as pd
import shap

model = joblib.load("diabetesPredictor.pkl")
explainer = shap.TreeExplainer(model)
feature_names = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]


def predict_health(features):
    input_df = pd.DataFrame(features, index=[0])
    input_df = input_df[feature_names]
    pred = model.predict(input_df)
    return pred[0], input_df


def get_explanation(input_df):
    shap_values = explainer(input_df)
    if shap_values.values.ndim == 3 and shap_values.values.shape[2] == 2:
        values = shap_values.values[0, :, 1]
    else:
        values = shap_values.values[0]
        if values.ndim > 1:
            values = values.flatten()
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "SHAP_Value": values}
    ).sort_values(by="SHAP_Value", ascending=False)
    return importance_df


st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🍬", layout="wide")
st.title("EarlyDetect/diabetes")
st.write("Enter your health metrics below to assess your diabetes risk.")
with st.form("input_form"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox(
            "Gender",
            options=[(0, "Male"), (1, "Female"), (2, "Other")],
            format_func=lambda x: x[1],
        )[0]
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    with col2:
        hypertension = st.selectbox(
            "Hypertension", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1]
        )[0]
        heart_disease = st.selectbox(
            "Heart Disease", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1]
        )[0]
    st.subheader("Health Metrics")
    col3, col4 = st.columns(2)
    with col3:
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        HbA1c_level = st.number_input(
            "HbA1c Level", min_value=3.0, max_value=15.0, value=5.5
        )
    with col4:
        blood_glucose_level = st.number_input(
            "Blood Glucose Level", min_value=50, max_value=300, value=100
        )
        smoking_history = st.selectbox(
            "Smoking History",
            options=[
                (0, "not current"),
                (1, "former"),
                (2, "never"),
                (3, "current"),
                (4, "No Info"),
                (5, "ever"),
            ],
            format_func=lambda x: x[1],
        )[0]
    submit_button = st.form_submit_button(label="Analyze Diabetes Risk")
if submit_button:
    features = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level,
    }
    result, input_df = predict_health(features)
    st.markdown("---")
    if result == 1:
        st.error("### Prediction: High risk of diabetes.")
        st.write(
            "Based on the provided metrics, there is a higher probability of diabetes. Please consult a healthcare professional for a detailed evaluation."
        )
    else:
        st.success("### Prediction: Low risk of diabetes.")
        st.write(
            "Based on the provided metrics, there is a lower probability of diabetes. Continue maintaining a healthy lifestyle."
        )
    st.markdown("---")
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.subheader("Your Input Summary")
        st.table(input_df.T.rename(columns={0: "Value"}))
    with col_r2:
        st.subheader("Feature Impact (SHAP Values)")
        st.write("Positive values increase risk, while negative values decrease risk.")
        explanation_df = get_explanation(input_df)
        st.bar_chart(explanation_df.set_index("Feature"))
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; padding: 20px;">
    <p>Made by bruzz-bruzz</p>
    <div style="display: flex; justify-content: center; gap: 20px;">
        <a href="https://github.com/bruzz-bruzz" target="_blank">GitHub Profile</a>
        <a href="https://github.com/bruzz-bruzz/cardiovascular-and-diabetic-health-predictor" target="_blank">GitHub Repository</a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
