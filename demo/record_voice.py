import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(duration=5, filename="recorded.wav"):
    fs = 16000

    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    write(filename, fs, recording)

    return filename