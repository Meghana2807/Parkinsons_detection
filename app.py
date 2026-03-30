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

from inference.predict import model
from inference.gradcam import generate_gradcam
from demo.feature_extraction import extract_features


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Parkinson’s Detection", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>
Parkinson’s Disease Detection System
</h1>
""", unsafe_allow_html=True)

st.markdown("### 🎤 Voice-Based AI Healthcare System")
st.markdown("---")


# -----------------------------
# FUNCTIONS
# -----------------------------
def generate_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    os.makedirs("temp", exist_ok=True)
    spec_path = "temp/spectrogram.png"

    plt.figure(figsize=(6,3))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(spec_path)
    plt.close()

    return spec_path


def spectrogram_to_tensor(spec_path):
    image = Image.open(spec_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    return transform(image).unsqueeze(0)


# -----------------------------
# INPUT
# -----------------------------
st.subheader("📂 Input Voice")

uploaded_file = st.file_uploader("Upload Voice (.wav)", type=["wav"])
audio_bytes = st.audio_input("Record Live Voice")

audio_path = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name
    st.audio(uploaded_file)

elif audio_bytes is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes.read())
        audio_path = tmp.name
    st.audio(audio_bytes)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if audio_path:

    y, sr = librosa.load(audio_path, sr=16000)

    st.markdown("---")

    # -----------------------------
    # WAVEFORM + SPECTROGRAM
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Voice Waveform")
        fig, ax = plt.subplots(figsize=(6,3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

    spec_path = generate_spectrogram(audio_path)

    with col2:
        st.subheader("🌈 Mel Spectrogram")
        st.image(spec_path)

    st.markdown("---")

    # -----------------------------
    # FEATURES
    # -----------------------------
    st.subheader("📊 Extracted Voice Parameters")

    try:
        features = extract_features(audio_path)

        c1, c2, c3 = st.columns(3)
        c1.metric("Pitch", round(features.get("pitch",0),2))
        c2.metric("Energy", round(features.get("energy",0),2))
        c3.metric("MFCC", round(features.get("mfcc",0),2))

    except:
        st.warning("Feature extraction failed")

    st.markdown("---")

    # -----------------------------
    # MODEL PREDICTION (CORRECT)
    # -----------------------------
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

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------
    if pd_prob < 0.5:
        predicted_class = "Healthy Control"
        confidence = healthy_percent
    else:
        predicted_class = "Parkinson’s Disease"
        confidence = pd_percent

    # -----------------------------
    # STAGE CLASSIFICATION
    # -----------------------------
    if predicted_class == "Healthy Control":
        stage = "None"
    elif confidence < 60:
        stage = "Early Stage"
    elif confidence < 80:
        stage = "Moderate Stage"
    else:
        stage = "Severe Stage"

    # -----------------------------
    # DISPLAY RESULT
    # -----------------------------
    st.subheader("🧠 Prediction Result")

    if predicted_class == "Healthy Control":
        st.success("✅ Healthy Control")
    else:
        st.error("⚠ Parkinson’s Disease Detected")

    st.write(f"**Confidence:** {confidence}%")

    if stage == "Early Stage":
        st.info("🟢 Early Stage")
    elif stage == "Moderate Stage":
        st.warning("🟠 Moderate Stage")
    elif stage == "Severe Stage":
        st.error("🔴 Severe Stage - Immediate Attention Required")
    else:
        st.success("No Disease Detected")

    st.markdown("---")

    # -----------------------------
    # PROBABILITY GRAPH (FIXED)
    # -----------------------------
    st.subheader("📉 Prediction Probability")

    df = pd.DataFrame({
        "Class": ["Healthy", "Parkinson"],
        "Probability": [healthy_percent, pd_percent]
    })

    st.bar_chart(df.set_index("Class"))

    st.markdown("---")

    # -----------------------------
    # GRAD-CAM (SMALL SIZE)
    # -----------------------------
    st.subheader("🔥 Model Attention Heatmap")

    cam = generate_gradcam(model, image_tensor)
    cam = cv2.resize(cam, (200,200))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    spec_img = cv2.imread(spec_path)
    spec_img = cv2.resize(spec_img, (200,200))

    overlay = cv2.addWeighted(spec_img, 0.6, heatmap, 0.4, 0)

    st.image(overlay, width=300)

    st.markdown("---")

    # -----------------------------
    # HOSPITAL SUGGESTION
    # -----------------------------
    if stage == "Severe Stage":
        st.subheader("🏥 Recommended Action")
        st.write("Visit a nearby neurologist immediately.")
        st.markdown("[Find Nearby Hospitals](https://www.google.com/maps/search/neurologist+near+me)")

    st.markdown("---")


# -----------------------------
# FOOTER
# -----------------------------
st.info("⚠ This system assists in Parkinson’s detection using voice. Not a medical diagnosis tool.")