import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# INPUT FOLDERS
HC_AUDIO_DIR = "datasets/Figshare_hc"
PD_AUDIO_DIR = "datasets/Figshare_pd"

# OUTPUT FOLDERS
HC_OUTPUT_DIR = "spectrograms/HC"
PD_OUTPUT_DIR = "spectrograms/PD"

os.makedirs(HC_OUTPUT_DIR, exist_ok=True)
os.makedirs(PD_OUTPUT_DIR, exist_ok=True)


def generate_and_save_spectrogram(audio_path, output_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis=None,
            y_axis=None
        )
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(output_path, dpi=150)
        plt.close()

    except Exception as e:
        print(f" Error processing {audio_path}: {e}")


# PROCESS HC FILES
print(" Processing Healthy Control audios...")
for file in os.listdir(HC_AUDIO_DIR):
    if file.lower().endswith(".wav"):
        input_path = os.path.join(HC_AUDIO_DIR, file)
        output_path = os.path.join(
            HC_OUTPUT_DIR,
            file.replace(".wav", ".png")
        )
        generate_and_save_spectrogram(input_path, output_path)

print(" HC spectrograms generated\n")


# PROCESS PD FILES
print(" Processing Parkinson’s Disease audios...")
for file in os.listdir(PD_AUDIO_DIR):
    if file.lower().endswith(".wav"):
        input_path = os.path.join(PD_AUDIO_DIR, file)
        output_path = os.path.join(
            PD_OUTPUT_DIR,
            file.replace(".wav", ".png")
        )
        generate_and_save_spectrogram(input_path, output_path)

print(" PD spectrograms generated\n")
print(" ALL spectrograms saved successfully.")
