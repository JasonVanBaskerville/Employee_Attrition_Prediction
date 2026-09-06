import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👤",
    layout="centered"
)


# ============================================================
# LOAD MODEL & THRESHOLD
# ============================================================

MODEL_PATH = "logistic_regression_pipeline.pkl"
THRESHOLD_PATH = "logistic_regression_threshold.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_threshold():
    return joblib.load(THRESHOLD_PATH)


try:
    model = load_model()
    threshold = load_threshold()

except FileNotFoundError:
    st.error(
        "Model atau file threshold tidak ditemukan. "
        "Pastikan kedua file berada di folder yang sama dengan app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca model: {e}")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("👤 Employee Attrition Prediction")

st.write(
    "Enter employee information to predict whether "
    "the employee is at risk of attrition."
)

st.divider()


# ============================================================
# INPUT FORM
# ============================================================

with st.form("attrition_form"):

    st.subheader("Employee Profil")

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    with col2:
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )

    # --------------------------------------------------------
    # JOB INFORMATION
    # --------------------------------------------------------

    st.subheader("Job Information")

    col1, col2 = st.columns(2)

    with col1:
        overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"]
        )

    with col2:
        job_level = st.number_input(
            "Job Level",
            min_value=1,
            max_value=5,
            value=1,
            step=1
        )

    monthly_income_bracket = st.selectbox(
        "Monthly Income Bracket",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    st.subheader("Income")

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000,
        step=500
    )

    # --------------------------------------------------------
    # WORK EXPERIENCE
    # --------------------------------------------------------

    st.subheader("Work Experience")

    col1, col2 = st.columns(2)

    with col1:
        total_working_years = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=60,
            value=5,
            step=1
        )

    with col2:
        years_at_company = st.number_input(
            "Years at Company",
            min_value=0,
            max_value=60,
            value=3,
            step=1
        )

    # --------------------------------------------------------
    # CURRENT POSITION
    # --------------------------------------------------------

    st.subheader("Current Position")

    col1, col2 = st.columns(2)

    with col1:
        years_in_current_role = st.number_input(
            "Years in Current Role",
            min_value=0,
            max_value=60,
            value=2,
            step=1
        )

    with col2:
        years_with_curr_manager = st.number_input(
            "Years with Current Manager",
            min_value=0,
            max_value=60,
            value=2,
            step=1
        )

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict Attrition",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "OverTime": [overtime],
        "JobRole": [job_role],
        "MonthlyIncomeBracket": [monthly_income_bracket],
        "TotalWorkingYears": [total_working_years],
        "MonthlyIncome": [monthly_income],
        "YearsAtCompany": [years_at_company],
        "JobLevel": [job_level],
        "YearsInCurrentRole": [years_in_current_role],
        "YearsWithCurrManager": [years_with_curr_manager],
        "Age": [age]
    })

    # --------------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------------

    try:

        probability = model.predict_proba(input_data)[0, 1]

        # Apply custom threshold
        if probability >= threshold:
            prediction = "Yes"
        else:
            prediction = "No"

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("Prediction Result")

        if prediction == "Yes":

            st.error(
                "⚠️ Attrition: **Yes**"
            )

        else:

            st.success(
                "✅ Attrition: **No**"
            )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat melakukan prediksi: {e}"
        )

