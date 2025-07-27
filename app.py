import streamlit as st
import joblib
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

#Load Models and Data
prediction_generated = False
with open("metrics.json", "r") as f:
    metrics = json.load(f)
model = joblib.load('best_salary_model.joblib')
label_encoder_gender = joblib.load('gender_encoder.joblib')
label_encoder_education = joblib.load('education_encoder.joblib')
label_encoder_job = joblib.load('job_title_encoder.joblib')
age_scaler = joblib.load('age_scaler.joblib')
exp_scaler = joblib.load('exp_scaler.joblib')

results_df = pd.read_csv("results_df.csv")

#Styles Configuration
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: white;
    }

    h1, h2, h3, h4 {
        color: white;
    }

    /* Predict Salary Button with your purple */
    div.stButton > button {
        background-color: #6A5ACD; /* Your purple */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }

    /* Slider label and selectbox label text */
    div[data-testid="stSlider"] label,
    .stSelectbox label {
        color: white;
    }

    /* Section Headings */
    .stMarkdown h2 {
        color: #00FFAA;
    }

    /* Success box */
    .stSuccess {
        background-color: #1c4133;
        border-radius: 6px;
        padding: 10px;
    }

    /* Divider styling */
    .divider {
        border-left: 2px solid #444;
        height: 100%;
        margin: auto;
    }

    /* Expander header font and hover */
    div[data-testid="stExpander"] details summary p {
        font-size: 22px;
        font-weight: bold;
        color: white;
   }
    [data-testid="stExpander"] details:hover summary p {
        color: #6A5ACD;
    }
            

    /* Glossary box styling */
    .glossary-box {
        background-color: #0b0b0bff;
        border-left: 5px solid #6A5ACD;
        padding: 1em;
        font-size: 17px;
        line-height: 1.6;
        color: white;
    }

    .term {
        color: #6A5ACD;
        font-weight: bold;
    }

    /* Slider thumb and track (WebKit browsers) */
    input[type="range"]::-webkit-slider-thumb {
        background: #6A5ACD;
    }

    input[type="range"]::-webkit-slider-runnable-track {
        background: #6A5ACD;
    }

    /* For Firefox */
    input[type="range"]::-moz-range-thumb {
        background: #6A5ACD;
    }

    input[type="range"]::-moz-range-track {
        background: #6A5ACD;
    }
</style>
""", unsafe_allow_html=True)


# Title
st.title("💼 Employee Salary Prediction App")
st.markdown("---") #Divider(line)

#layout: two columns with a divider
left_col, divider, right_col = st.columns([1, 0.05, 1])

with left_col:
    st.header("📝 Provide Your Details")
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    job_title = st.selectbox("Job Title", list(label_encoder_job.classes_))
    experience = st.slider("Years of Experience", 0, 40, 5)
    predict = st.button("🔮 Predict Salary")

with divider:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

with right_col:
    st.header("📊 Prediction Output")
    if predict:
        try:
            gender_encoded = label_encoder_gender.transform([gender])[0]
            education_encoded = label_encoder_education.transform([education])[0]
            job_encoded = label_encoder_job.transform([job_title])[0]
        except ValueError as e:
            st.error(f"Input error: {e}")
            st.stop()

        age_scaled = age_scaler.transform([[age]])[0][0]
        experience_scaled = exp_scaler.transform([[experience]])[0][0]

        input_data = np.array([[age_scaled, gender_encoded, education_encoded, job_encoded, experience_scaled]])
        salary_usd = model.predict(input_data)[0]

        st.success(f"### 💰 Estimated Annual Salary: ${salary_usd:,.2f}")

        prediction_generated = True

        # Performance Summary
        with st.container():
            st.markdown("### ✅ Model Performance Summary")
            st.markdown(
                f"""
                <div style="background-color:#1e1e2f; padding: 20px; border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0,255,170,0.2); color: #FAFAFA;
                            font-size: 17px; line-height: 1.8;">
                    <p><b>📈 R² Score:</b> {metrics['r2']:.4f}</p>
                    <p><b>✅ Accuracy:</b> {metrics['accuracy']:.2f}%</p>
                    <p><b>🔢 Mean Squared Error (MSE):</b> {metrics['mse']:.2f}</p>
                    <p><b>📉 Root Mean Squared Error (RMSE):</b> {metrics['rmse']:.2f}</p>
                    <p><b>📊 Mean Absolute Error (MAE):</b> {metrics['mae']:.2f}</p>
                    <p><b>🕒 Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

#Results Graph

#Sorting models by Accuracy 
best_model = results_df.sort_values(by="Accuracy (%)", ascending=False).iloc[0]


# Displaying the results
st.markdown("---")
if prediction_generated:
    st.markdown("## 🔍 Model Evaluation")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot: Accuracy %
        fig = px.bar(results_df, x='Model', y='Accuracy (%)',
                     color='Model',
                     title='Model Comparison (Higher Accuracy is Better)',
                     color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Model RMSE Summary")
        for index, row in results_df.iterrows():
            st.markdown(f"- **{row['Model']}**: RMSE = `{row['RMSE']:.2f}`")
        
        st.markdown("✅ **After evaluating all models, the best performing model is:**")
        st.markdown(f"### 🎯 `{best_model['Model']}`")

    st.markdown("---")

#Evaluation Metrics Explanation
with st.expander("📊 Evaluation Metrics – Explained"):
    st.markdown("""
    <div class="glossary-box">
    <p><span class="term">MAE (Mean Absolute Error):</span>  
    MAE calculates the average of the absolute differences between the predicted and actual values. 
    Example: ₹5,000 MAE → Predictions are off by ₹5,000 on average.</p>

    <p><span class="term">MSE (Mean Squared Error):</span>  
    MSE calculates the average of the squared differences between the predicted and actual values.</p>

    <p><span class="term">RMSE (Root Mean Squared Error):</span>  
    Square root of MSE, it’s in the same units as the target variable. Focuses on big errors.</p>

    <p><span class="term">R² Score (Coefficient of Determination):</span>  
    Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
    R² = 0.9 means 90% of salary variation is explained by the model.</p>

    <p><span class="term">Accuracy (%):</span>  
    (If used) Tells us the proportion of correct predictions made by the model out of all predictions.</p>
    </div>
    """, unsafe_allow_html=True)