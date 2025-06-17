import torch
import torchaudio
import joblib
import numpy as np
import pandas as pd
import os
from transformers import ClapProcessor, ClapModel


clf = joblib.load("mlp_classifier.joblib")
le = joblib.load("label_encoder.joblib")

test_df = pd.read_csv("test_files.csv")

processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
model.eval()

results = []

for idx, row in test_df.iterrows():
    filename = row["filename"]
    true_label = row["category"]
    file_path = os.path.join("audio", filename)

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

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
        correct = int(pred_label == true_label)

        results.append({
            "filename": filename,
            "true_category": true_label,
            "predicted_category": pred_label,
            "confidence": round(confidence, 2),
            "correct": correct
        })

        print(f"[{idx+1}/{len(test_df)}] {filename} — True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2f}, Correct: {correct}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")


results_df = pd.DataFrame(results)
results_df.to_csv("test_results.csv", index=False)
print("\nSaved test_results.csv")


if len(results) > 0:
    acc = results_df["correct"].mean()
    print(f"\nOverall Accuracy: {acc:.2%}")


