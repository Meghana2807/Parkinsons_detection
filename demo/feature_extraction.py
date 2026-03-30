import librosa
import numpy as np


def extract_features(audio_path):

    y, sr = librosa.load(audio_path, sr=16000)

    # -------- Pitch --------
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])

    # -------- Energy --------
    energy = np.mean(librosa.feature.rms(y=y))

    # -------- MFCC --------
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

    features = {
        "pitch": float(pitch),
        "energy": float(energy),
        "mfcc": float(mfcc)
    }

    return features