import streamlit as st
import pandas as pd
import joblib

# Path to the saved model
MODEL_PATH = 'attrition_model.pkl'

# Page Title
st.title("Employee Attrition Prediction App")
st.markdown("Enter employee details below to predict attrition probability.")

# Function to take user input
def user_input():
    Age = st.slider("Age", 18, 60, 30)
    DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
    Education = st.selectbox("Education (1=Below College, 5=Doctor)", [1, 2, 3, 4, 5])
    JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    MonthlyIncome = st.slider("Monthly Income", 1000, 20000, 5000)
    NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 1)
    PercentSalaryHike = st.slider("Percent Salary Hike", 0, 100, 15)
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
    TrainingTimesLastYear = st.slider("Trainings Last Year", 0, 10, 2)
    YearsAtCompany = st.slider("Years at Company", 0, 40, 5)

    data = {
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'Education': Education,
        'JobLevel': JobLevel,
        'MonthlyIncome': MonthlyIncome,
        'NumCompaniesWorked': NumCompaniesWorked,
        'PercentSalaryHike': PercentSalaryHike,
        'TotalWorkingYears': TotalWorkingYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'YearsAtCompany': YearsAtCompany
    }
    return pd.DataFrame([data])

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.stop()

# Get input from user
input_df = user_input()

# Predict on button click
if st.button("Predict Attrition", key="predict_button"):
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]  # Probability of class '1' (attrition)

        # Show result
        if prediction[0] == 1:
            st.error("⚠️ This employee is likely to leave the company.")
        else:
            st.success("✅ This employee is likely to stay.")

        st.markdown(f"**Prediction Confidence:** {probability:.2%}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
