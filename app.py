import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

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
    # Clinical Data Inputs (Matching Model Expectation)
    st.header("Clinical Data")
    age = st.slider("Age", 30, 90, 60)
    bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, step=0.1)
    sleep_quality = st.slider("Sleep Quality", 0.0, 10.0, 5.0)
    tremor = st.selectbox("Tremor", [0, 1])
    rigidity = st.selectbox("Rigidity", [0, 1])
    bradykinesia = st.selectbox("Bradykinesia", [0, 1])
    postural_instability = st.selectbox("Postural Instability", [0, 1])
    
    # Additional features to match model expectation (placeholders)
    smoking = st.selectbox("Smoking", [0, 1])
    alcohol_consumption = st.selectbox("Alcohol Consumption", [0, 1])
    physical_activity = st.slider("Physical Activity Level", 0, 10, 5)
    diet_quality = st.slider("Diet Quality Score", 0, 10, 5)

    # Voice Data Inputs (Matching Model Expectation)
    st.header("Voice Data")
    mdvp_fo = st.slider("MDVP:Fo (Hz)", 80.0, 300.0, step=0.1)
    mdvp_fhi = st.slider("MDVP:Fhi (Hz)", 100.0, 600.0, step=0.1)
    mdvp_flo = st.slider("MDVP:Flo (Hz)", 50.0, 300.0, step=0.1)
    mdvp_jitter = st.slider("MDVP:Jitter (%)", 0.0, 1.0, step=0.01)
    mdvp_shimmer = st.slider("MDVP:Shimmer (%)", 0.0, 1.0, step=0.01)

    if st.button("Predict"):
        # Prepare input features
        clinical_features = np.array([[age, bmi, sleep_quality, tremor, rigidity, bradykinesia, postural_instability,
                                       smoking, alcohol_consumption, physical_activity, diet_quality]])
        
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
st.markdown("Developed with using Streamlit")
