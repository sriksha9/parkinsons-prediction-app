import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Parkinson's Disease Detector", layout="wide")

# Load trained models
def load_model(filename):
    try:
        return pickle.load(open(filename, "rb"))
    except FileNotFoundError:
        st.error(f"Error: The model file '{filename}' is missing. Ensure it exists in the directory.")
        return None

model_clinical = load_model("parkinsons_clinical_model.pkl")
model_voice = load_model("parkinsons_voice_model.pkl")

# Streamlit UI Setup
st.set_page_config(page_title="Parkinson's Disease Detector", layout="wide")
st.title("Parkinson's Disease Prediction")
st.write("Enter the required details to check for Parkinson’s Disease.")

if model_clinical is None or model_voice is None:
    st.error("Error: One or both model files are missing. Ensure they exist.")
    st.stop()  # Stop execution if models are missing
else:
    # Clinical Data Inputs
    st.header("Clinical Data")
    age = st.slider("Age", 30, 90, 60)
    bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, step=0.1)
    sleep_quality = st.slider("Sleep Quality", 0.0, 10.0, 5.0)
    tremor = st.selectbox("Tremor", [0, 1])
    rigidity = st.selectbox("Rigidity", [0, 1])
    bradykinesia = st.selectbox("Bradykinesia", [0, 1])
    postural_instability = st.selectbox("Postural Instability", [0, 1])

    # Voice Data Inputs
    st.header("Voice Data")
    jitter = st.slider("Jitter (%)", 0.0, 1.0, step=0.01)
    shimmer = st.slider("Shimmer (%)", 0.0, 1.0, step=0.01)
    hnr = st.slider("Harmonics-to-Noise Ratio (HNR)", 0.0, 40.0, step=0.1)

    if st.button("Predict"):
        # Prepare input features
        clinical_features = np.array([[age, bmi, sleep_quality, tremor, rigidity, bradykinesia, postural_instability]])
        voice_features = np.array([[jitter, shimmer, hnr]])
        
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
st.markdown("Developed with love using Streamlit")
