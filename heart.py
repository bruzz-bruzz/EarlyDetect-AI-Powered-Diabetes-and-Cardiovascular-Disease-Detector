import streamlit as st
import joblib
import pandas as pd
import shap
model = joblib.load('heartPredictor.pkl')
explainer = shap.TreeExplainer(model)

feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

def predict_health(features):
    input_df = pd.DataFrame([features])
    input_df = input_df[feature_names]
    pred = model.predict(input_df)
    return pred[0], input_df

def get_explanation(input_df):
    shap_values = explainer(input_df)
    values = shap_values.values[0]
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": values
    }).sort_values(by="SHAP_Value", ascending=False)
    
    return importance_df

st.set_page_config(page_title="Cardiovascular Health Predictor", page_icon="💖", layout="wide")
st.title("💖 Cardiovascular Health Predictor")
st.write("Enter your health metrics below to assess your cardiovascular health risk.")

with st.form("input_form"):
    st.subheader("Patient Information")
    with st.expander("Expand to enter basic information"):
        col1, col2 = st.columns(2)
        with col1:
            age_years = st.number_input('Age (years)', min_value=18, max_value=100, value=30, help="Age in years.")
            gender = st.selectbox('Gender', options=[(1, 'Male'), (2, 'Female')], format_func=lambda x: x[1], help="Gender of the patient.")[0]
        with col2:
            height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170, help="Height in centimeters.")
            weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70, help="Weight in kilograms.")
    
    st.subheader("Health Metrics")
    with st.expander("Expand to enter blood and health metrics"):
        col3, col4 = st.columns(2)
        with col3:
            ap_hi = st.number_input('Systolic Blood Pressure (ap_hi)', min_value=50, max_value=250, value=120, help="Systolic blood pressure.")
            ap_lo = st.number_input('Diastolic Blood Pressure (ap_lo)', min_value=30, max_value=150, value=80, help="Diastolic blood pressure.")
        with col4:
            cholesterol = st.selectbox('Cholesterol', options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1], help="Cholesterol level.")[0]
            gluc = st.selectbox('Glucose', options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1], help="Glucose level.")[0]

    st.subheader("Lifestyle")
    with st.expander("Expand to enter lifestyle habits"):
        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.selectbox('Smoke', options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1], help="Do you smoke?")[0]
        with col6:
            alco = st.selectbox('Alcohol Intake', options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1], help="Do you consume alcohol?")[0]
        with col7:
            active = st.selectbox('Physically Active', options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1], help="Are you physically active?")[0]
    
    submit_button = st.form_submit_button(label="Analyze Cardiovascular Risk")

if submit_button:
    features = {
        'age': age_years * 365,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }
    
    result, input_df = predict_health(features)
    
    st.markdown("---")
    if result == 1:
        st.error("### Prediction: High risk of heart disease.")
        st.write("Based on the provided metrics, there is a higher probability of cardiovascular disease. Please consult a healthcare professional for a detailed evaluation.")
    else:
        st.success("### Prediction: Low risk of heart disease.")
        st.write("Based on the provided metrics, there is a lower probability of cardiovascular disease. Continue maintaining a healthy lifestyle.")
    
    st.markdown("---")
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.subheader("Your Input Summary")
        input_display = input_df.copy()
        input_display['age'] = age_years
        st.table(input_display.T.rename(columns={0: 'Value'}))
        
    with col_r2:
        st.subheader("Feature Impact (SHAP Values)")
        st.write("Positive values increase risk, while negative values decrease risk.")
        explanation_df = get_explanation(input_df)
        st.bar_chart(explanation_df.set_index('Feature'))

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Made by bruzz-bruzz</p>
    <div style="display: flex; justify-content: center; gap: 20px;">
        <a href="https://github.com/bruzz-bruzz" target="_blank">GitHub Profile</a>
        <a href="https://github.com/bruzz-bruzz/cardiovascular-and-diabetic-health-predictor" target="_blank">GitHub Repository</a>
    </div>
</div>
""", unsafe_allow_html=True)