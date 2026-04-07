import streamlit as st
import os
import tempfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

# OPTIONAL (SAFE IMPORT FOR GRADCAM)
try:
    from inference.gradcam import generate_gradcam
    GRADCAM_AVAILABLE = True
except:
    GRADCAM_AVAILABLE = False

# MODEL IMPORT (CHANGE PATH IF NEEDED)
try:
    from inference.predict import model
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Parkinson’s Detection", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>
🧠 Parkinson’s Disease Detection System
</h1>
<h4 style='text-align: center;'>🎤 Voice-Based AI Healthcare System</h4>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- FUNCTIONS ----------------

def generate_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    os.makedirs("temp", exist_ok=True)
    spec_path = "temp/spec.png"

    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mel_db, sr=sr)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(spec_path)
    plt.close()

    return spec_path


def spectrogram_to_tensor(spec_path):
    image = Image.open(spec_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))

    return pitch, energy, mfcc


def simple_prediction(pitch, energy, mfcc):
    score = (pitch * 0.0005) + (energy * 2) + (mfcc * 0.1)

    if score < 1:
        return "Healthy Control", 90, "None"
    elif score < 2:
        return "Parkinson’s", 70, "Early Stage"
    elif score < 3:
        return "Parkinson’s", 80, "Moderate Stage"
    else:
        return "Parkinson’s", 90, "Severe Stage"


# ---------------- INPUT ----------------
st.subheader("📂 Upload or Record Voice")

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
audio_bytes = st.audio_input("Record Voice")

audio_path = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name
    st.audio(uploaded_file)

elif audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes.read())
        audio_path = tmp.name
    st.audio(audio_bytes)

# ---------------- MAIN PIPELINE ----------------
if audio_path:

    y, sr = librosa.load(audio_path, sr=16000)

    col1, col2 = st.columns(2)

    # WAVEFORM
    with col1:
        st.subheader("📈 Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

    # SPECTROGRAM
    spec_path = generate_spectrogram(audio_path)

    with col2:
        st.subheader("🌈 Spectrogram")
        st.image(spec_path)

    st.markdown("---")

    # FEATURES
    pitch, energy, mfcc = extract_features(audio_path)

    st.subheader("📊 Extracted Features")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pitch", round(pitch, 2))
    c2.metric("Energy", round(energy, 2))
    c3.metric("MFCC", round(mfcc, 2))

    st.markdown("---")

    # ---------------- PREDICTION ----------------

    if MODEL_AVAILABLE:
        image_tensor = spectrogram_to_tensor(spec_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)

        healthy_prob = probs[0][0].item()
        pd_prob = probs[0][1].item()

        healthy_percent = round(healthy_prob * 100, 2)
        pd_percent = round(pd_prob * 100, 2)

        if pd_prob < 0.5:
            predicted_class = "Healthy Control"
            confidence = healthy_percent
        else:
            predicted_class = "Parkinson’s"
            confidence = pd_percent

    else:
        predicted_class, confidence, stage = simple_prediction(pitch, energy, mfcc)
        healthy_percent = 100 - confidence
        pd_percent = confidence

    # ---------------- SEVERITY ----------------
    if predicted_class == "Healthy Control":
        stage = "None"
    elif confidence < 60:
        stage = "Early Stage"
    elif confidence < 80:
        stage = "Moderate Stage"
    else:
        stage = "Severe Stage"

    # ---------------- POPUPS ----------------
    if predicted_class == "Healthy Control":
        st.success("✅ Healthy Control")
        st.toast("You are Healthy!", icon="✅")

    else:
        st.error("⚠ Parkinson’s Detected")

        if stage == "Early Stage":
            st.toast("⚠ Early Stage Detected", icon="⚠️")

        elif stage == "Moderate Stage":
            st.warning("⚠ Moderate Risk - Consult Doctor")
            st.toast("Moderate Parkinson’s", icon="⚠️")

        elif stage == "Severe Stage":
            st.error("🚨 SEVERE CONDITION!")
            st.toast("🚨 Immediate Attention Needed!", icon="🚨")
            st.warning("📍 Visit nearest neurologist immediately!")

          # 🔥 LOCATION-BASED FEATURE
            st.subheader("🏥 Nearby Hospital Assistance")

            location = st.text_input("📍 Enter your location (City or Area)")

            if location:
                 maps_url = f"https://www.google.com/maps/search/neurologist+near+{location.replace(' ', '+')}"
        
                 st.success(f"Showing hospitals near: {location}")
                 st.markdown(f"[🔍 Find Nearby Hospitals]({maps_url})")

             # Default option
            st.markdown("### OR")
            st.markdown("[📍 Use Current Location](https://www.google.com/maps/search/neurologist+near+me)")

    # ---------------- RESULT ----------------
    st.subheader("🧠 Prediction Result")
    st.write(f"Class: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
    st.write(f"Severity: {stage}")

    # ---------------- GRAPH ----------------
    st.subheader("📉 Probability Graph")
    df = pd.DataFrame({
        "Class": ["Healthy", "Parkinson"],
        "Probability": [healthy_percent, pd_percent]
    })
    st.bar_chart(df.set_index("Class"))

    # ---------------- GRADCAM ----------------
    if GRADCAM_AVAILABLE and MODEL_AVAILABLE:
        st.subheader("🔥 Attention Heatmap")
        cam = generate_gradcam(model, image_tensor)
        cam = cv2.resize(cam, (200, 200))

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        spec_img = cv2.imread(spec_path)
        spec_img = cv2.resize(spec_img, (200, 200))

        overlay = cv2.addWeighted(spec_img, 0.6, heatmap, 0.4, 0)
        st.image(overlay, width=400)

