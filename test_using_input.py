import torch
import torchaudio
import joblib
import numpy as np
import os
from transformers import ClapProcessor, ClapModel

clf = joblib.load("mlp_classifier.joblib")
le = joblib.load("label_encoder.joblib")

processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
model.eval()

filename = input("Enter the audio filename (e.g., 1-100032-A-0.wav): ").strip()
file_path = os.path.join("audio", filename)

if not os.path.exists(file_path):
    print(f"File does not exist: {file_path}")
    exit(1)

try:
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0).unsqueeze(0)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(waveform)
    waveform = waveform[:, :480000]

    with torch.no_grad():
        inputs = processor(audios=waveform.squeeze().numpy(), sampling_rate=48000, return_tensors="pt")
        emb = model.get_audio_features(**inputs).squeeze().numpy()

    probs = clf.predict_proba([emb])[0]
    pred_index = np.argmax(probs)
    pred_label = le.inverse_transform([pred_index])[0]
    confidence = probs[pred_index]

    print(f"\nPrediction for '{filename}':")
    print(f"Predicted Category: {pred_label}")
    print(f"Confidence: {confidence:.2f}")

except Exception as e:
    print(f"Error processing file: {e}")
