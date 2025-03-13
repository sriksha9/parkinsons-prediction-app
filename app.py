{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abb83dfc-87da-4fc6-b6e3-8146ac47b14f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load trained models\n",
    "def load_model(filename):\n",
    "    try:\n",
    "        return pickle.load(open(filename, \"rb\"))\n",
    "    except FileNotFoundError:\n",
    "        st.error(f\"Error: The model file '{filename}' is missing. Ensure it exists in the directory.\")\n",
    "        return None\n",
    "\n",
    "model_clinical = load_model(\"parkinsons_clinical_model.pkl\")\n",
    "model_voice = load_model(\"parkinsons_voice_model.pkl\")\n",
    "\n",
    "# Streamlit UI Setup\n",
    "st.set_page_config(page_title=\"Parkinson's Disease Detector\", layout=\"wide\")\n",
    "st.title(\"Parkinson's Disease Prediction\")\n",
    "st.write(\"Enter the required details to check for Parkinson’s Disease.\")\n",
    "\n",
    "if model_clinical is None or model_voice is None:\n",
    "    st.error(\"Error: One or both model files are missing. Ensure they exist.\")\n",
    "    st.stop()  # Stop execution if models are missing\n",
    "else:\n",
    "    # Clinical Data Inputs\n",
    "    st.header(\"Clinical Data\")\n",
    "    age = st.slider(\"Age\", 30, 90, 60)\n",
    "    bmi = st.number_input(\"BMI\", min_value=10.0, max_value=40.0, step=0.1)\n",
    "    sleep_quality = st.slider(\"Sleep Quality\", 0.0, 10.0, 5.0)\n",
    "    tremor = st.selectbox(\"Tremor\", [0, 1])\n",
    "    rigidity = st.selectbox(\"Rigidity\", [0, 1])\n",
    "    bradykinesia = st.selectbox(\"Bradykinesia\", [0, 1])\n",
    "    postural_instability = st.selectbox(\"Postural Instability\", [0, 1])\n",
    "\n",
    "    # Voice Data Inputs\n",
    "    st.header(\"Voice Data\")\n",
    "    jitter = st.slider(\"Jitter (%)\", 0.0, 1.0, step=0.01)\n",
    "    shimmer = st.slider(\"Shimmer (%)\", 0.0, 1.0, step=0.01)\n",
    "    hnr = st.slider(\"Harmonics-to-Noise Ratio (HNR)\", 0.0, 40.0, step=0.1)\n",
    "\n",
    "    if st.button(\"Predict\"):\n",
    "        # Prepare input features\n",
    "        clinical_features = np.array([[age, bmi, sleep_quality, tremor, rigidity, bradykinesia, postural_instability]])\n",
    "        voice_features = np.array([[jitter, shimmer, hnr]])\n",
    "        \n",
    "        # Get Predictions\n",
    "        clinical_pred = model_clinical.predict(clinical_features)[0]\n",
    "        voice_pred = model_voice.predict(voice_features)[0]\n",
    "        \n",
    "        # Get confidence scores\n",
    "        clinical_proba = model_clinical.predict_proba(clinical_features)[0][1] * 100\n",
    "        voice_proba = model_voice.predict_proba(voice_features)[0][1] * 100\n",
    "        \n",
    "        # Final Diagnosis\n",
    "        if clinical_pred == 1 or voice_pred == 1:\n",
    "            diagnosis = \"Parkinson's Detected\"\n",
    "        else:\n",
    "            diagnosis = \"No Parkinson's Detected\"\n",
    "        \n",
    "        st.success(f\"Prediction: {diagnosis}\")\n",
    "        st.write(f\"Clinical Model Confidence: {clinical_proba:.2f}%\")\n",
    "        st.write(f\"Voice Model Confidence: {voice_proba:.2f}%\")\n",
    "\n",
    "# Footer\n",
    "st.markdown(\"Developed with love using Streamlit\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
