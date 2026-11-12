import streamlit as st
st.title("About EarlyDetect")
st.markdown("""
This project is a health risk assessment tool designed to help users assess their risk for both diabetes and cardiovascular disease. It utilizes machine learning models trained with Python, integrated into a user-friendly Streamlit web application.
- **Diabetes Risk Prediction**: Assesses diabetes risk based on clinical features such as age, gender, BMI, HbA1c levels, and blood glucose.
- **Cardiovascular Disease Risk Prediction**: Assesses cardiovascular disease risk based on clinical metrics like blood pressure, cholesterol, and lifestyle factors.
- **Explainability**: Uses SHAP (SHapley Additive exPlanations) to provide insights into which health metrics influenced each prediction, helping users understand the reasoning behind their results.
- **Python**: Core programming language.
- **Streamlit**: For the web interface.
- **XGBoost/CatBoost**: Machine learning models for predictions.
- **SHAP**: For model interpretability.
- This tool is for **educational and informational purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")
