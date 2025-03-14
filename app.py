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
        st.error(f" Error: The model file '{filename}' is missing. Upload it manually in Hugging Face Spaces.")
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

    # Voice Data Inputs
    st.header("Voice Data")
    mdvp_fo = st.slider("MDVP:Fo (Hz)", 80.0, 300.0, step=0.1)
    mdvp_fhi = st.slider("MDVP:Fhi (Hz)", 100.0, 600.0, step=0.1)
    mdvp_flo = st.slider("MDVP:Flo (Hz)", 50.0, 300.0, step=0.1)
    mdvp_jitter = st.slider("MDVP:Jitter (%)", 0.0, 1.0, step=0.01)
    mdvp_shimmer = st.slider("MDVP:Shimmer (%)", 0.0, 1.0, step=0.01)

    if st.button("Predict"):
        # Prepare input features
        clinical_features = np.array([[age, bmi, sleep_quality, tremor, rigidity, bradykinesia, postural_instability,
                                       gender == "Male", ethnicity == "White", education_level, smoking, 
                                       alcohol_consumption, physical_activity, diet_quality, family_history, 
                                       brain_injury, hypertension, diabetes, depression, stroke, systolic_bp, 
                                       diastolic_bp, cholesterol_total, cholesterol_ldl, cholesterol_hdl, cholesterol_triglycerides,
                                       updrs, moca, functional_assessment, speech_problems, sleep_disorders, constipation]])

        voice_features = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_shimmer]])

        # Debugging: Print expected features
        st.write(" Clinical Model expects:", model_clinical.n_features_in_, "features")
        st.write(" Voice Model expects:", model_voice.n_features_in_, "features")

        # Ensure input shape matches model expectation
        if clinical_features.shape[1] != model_clinical.n_features_in_:
            st.error(f" Error: Clinical model expects {model_clinical.n_features_in_} features, but received {clinical_features.shape[1]}.")
        elif voice_features.shape[1] != model_voice.n_features_in_:
            st.error(f" Error: Voice model expects {model_voice.n_features_in_} features, but received {voice_features.shape[1]}.")
        else:
            # Get Predictions
            clinical_pred = model_clinical.predict(clinical_features)[0]
            voice_pred = model_voice.predict(voice_features)[0]

            # Get confidence scores
            clinical_proba = model_clinical.predict_proba(clinical_features)[0][1] * 100
            voice_proba = model_voice.predict_proba(voice_features)[0][1] * 100

            # Final Diagnosis
            if clinical_pred == 1 or voice_pred == 1:
                diagnosis = "Parkinson's Detected"
            else:
                diagnosis = "No Parkinson's Detected"

            st.success(f"Prediction: {diagnosis}")
            st.write(f"Clinical Model Confidence: {clinical_proba:.2f}%")
            st.write(f"Voice Model Confidence: {voice_proba:.2f}%")

# Footer
st.markdown("Developed with  using Streamlit")
