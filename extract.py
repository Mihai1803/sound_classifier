import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import ClapProcessor, ClapModel

processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
model.eval()

df = pd.read_csv("meta/esc50.csv")
embeddings = []
labels = []

for idx, row in df.iterrows():
    path = os.path.join("audio", row["filename"])
    try:
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0).unsqueeze(0)  # convert to mono
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(waveform) # 48k fs
        waveform = waveform[:, :480000]  # crop to 10s

        with torch.no_grad():
            inputs = processor(audios=waveform.squeeze().numpy(), sampling_rate=48000, return_tensors="pt")
            emb = model.get_audio_features(**inputs)
            embeddings.append(emb.squeeze().numpy())
            labels.append(row["category"])

        print(f"[{idx+1}/{len(df)}] Processed: {row['filename']}")
    except Exception as e:
        print(f"Failed to process {path}: {e}")

np.save("embeddings.npy", np.stack(embeddings))
label_df = pd.DataFrame({
    "filename": df["filename"],
    "category": labels
})
label_df.to_csv("labels.csv", index=False)
print("Extracted CLAP embeddings.")
