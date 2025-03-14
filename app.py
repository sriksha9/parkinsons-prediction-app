import streamlit as st
import pickle
import numpy as np
import os

# Set Streamlit Page Configuration
st.set_page_config(page_title="Parkinson's Disease Detector", layout="wide")

# Function to load models
def load_model(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        st.error(f"  Error: The model file '{filename}' is missing. Upload it manually in Hugging Face Spaces.")
        return None

# Load models
model_clinical = load_model("parkinsons_clinical_model.pkl")
model_voice = load_model("parkinsons_voice_model.pkl")

# Streamlit UI Setup
st.title("Parkinson's Disease Prediction")
st.write("Enter the required details to check for Parkinson’s Disease.")

if model_clinical is None or model_voice is None:
    st.error("Error: One or both model files are missing. Upload them to Hugging Face Spaces.")
    st.stop()
else:
    # Clinical Data Inputs (All 32 Features)
    st.header("Clinical Data")
    age = st.slider("Age", 30, 90, 60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "White", "Other"])
    education_level = st.slider("Education Level (Years)", 0, 20, 12)
    bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, step=0.1)
    smoking = st.selectbox("Smoking", [0, 1])
    alcohol_consumption = st.selectbox("Alcohol Consumption", [0, 1])
    physical_activity = st.slider("Physical Activity Level", 0, 10, 5)
    diet_quality = st.slider("Diet Quality Score", 0, 10, 5)
    sleep_quality = st.slider("Sleep Quality", 0, 10, 5)
    family_history = st.selectbox("Family History of Parkinson’s", [0, 1])
    brain_injury = st.selectbox("Traumatic Brain Injury", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    depression = st.selectbox("Depression", [0, 1])
    stroke = st.selectbox("Stroke", [0, 1])
    systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
    cholesterol_total = st.number_input("Total Cholesterol", 100, 300, 180)
    cholesterol_ldl = st.number_input("LDL Cholesterol", 50, 200, 100)
    cholesterol_hdl = st.number_input("HDL Cholesterol", 20, 100, 50)
    cholesterol_triglycerides = st.number_input("Triglycerides", 50, 300, 150)

    updrs = st.slider("UPDRS Score", 0, 100, 50)
    moca = st.slider("MoCA Score", 0, 30, 15)
    functional_assessment = st.slider("Functional Assessment Score", 0, 100, 50)
    speech_problems = st.selectbox("Speech Problems", [0, 1])
    sleep_disorders = st.selectbox("Sleep Disorders", [0, 1])
    constipation = st.selectbox("Constipation", [0, 1])

    # Motor Symptoms
    tremor = st.selectbox("Tremor", [0, 1])
    rigidity = st.selectbox("Rigidity", [0, 1])
    bradykinesia = st.selectbox("Bradykinesia", [0, 1])
    postural_instability = st.selectbox("Postural Instability", [0, 1])

    # Voice Data Inputs (All 22 Features)
    st.header("Voice Data")
    mdvp_fo = st.slider("MDVP:Fo (Hz)", 80.0, 300.0, step=0.1)
    mdvp_fhi = st.slider("MDVP:Fhi (Hz)", 100.0, 600.0, step=0.1)
    mdvp_flo = st.slider("MDVP:Flo (Hz)", 50.0, 300.0, step=0.1)
    mdvp_jitter = st.slider("MDVP:Jitter (%)", 0.0, 1.0, step=0.01)
    mdvp_jitter_abs = st.slider("MDVP:Jitter(Abs)", 0.0, 0.1, step=0.001)
    mdvp_rap = st.slider("MDVP:RAP", 0.0, 0.2, step=0.001)
    mdvp_ppq = st.slider("MDVP:PPQ", 0.0, 0.2, step=0.001)
    jitter_ddp = st.slider("Jitter:DDP", 0.0, 0.2, step=0.001)

    mdvp_shimmer = st.slider("MDVP:Shimmer", 0.0, 1.0, step=0.01)
    mdvp_shimmer_db = st.slider("MDVP:Shimmer(dB)", 0.0, 2.0, step=0.01)
    shimmer_apq3 = st.slider("Shimmer:APQ3", 0.0, 0.5, step=0.001)
    shimmer_apq5 = st.slider("Shimmer:APQ5", 0.0, 0.5, step=0.001)
    mdvp_apq = st.slider("MDVP:APQ", 0.0, 0.5, step=0.001)
    shimmer_dda = st.slider("Shimmer:DDA", 0.0, 0.5, step=0.001)

    nhr = st.slider("NHR", 0.0, 1.0, step=0.01)
    hnr = st.slider("HNR", 0.0, 40.0, step=0.1)
    rpde = st.slider("RPDE", 0.0, 1.0, step=0.01)
    dfa = st.slider("DFA", 0.0, 1.0, step=0.01)
    spread1 = st.slider("Spread1", -7.0, -2.0, step=0.01)
    spread2 = st.slider("Spread2", 0.0, 1.0, step=0.01)
    d2 = st.slider("D2", 0.0, 4.0, step=0.01)
    ppe = st.slider("PPE", 0.0, 1.0, step=0.01)

    if st.button("Predict"):
        # Prepare input features
        clinical_features = np.array([[age, bmi, sleep_quality, tremor, rigidity, bradykinesia, postural_instability,
                                       cholesterol_total, cholesterol_ldl, cholesterol_hdl, cholesterol_triglycerides,
                                       updrs, moca, functional_assessment, speech_problems, sleep_disorders, constipation]])

        voice_features = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs, mdvp_rap, mdvp_ppq,
                                    jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq,
                                    shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

        # Predictions
        clinical_pred = model_clinical.predict(clinical_features)[0]
        voice_pred = model_voice.predict(voice_features)[0]

        diagnosis = "Parkinson's Detected" if clinical_pred == 1 or voice_pred == 1 else "No Parkinson's Detected"
        st.success(f"Prediction: {diagnosis}")

# Footer
st.markdown("Developed with  using Streamlit")
