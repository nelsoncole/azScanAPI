# pip install --user uvicorn
# pip install --user fastapi
# pip install --user "uvicorn[standard]"
# uvicorn FastAPI:app --reload --host 0.0.0.0 --port 8082
# python -m uvicorn FastAPI:app --reload --host 0.0.0.0 --port 8082

import torch
import torch.nn as nn
import numpy as np
import base64
from fastapi import FastAPI
from pydantic import BaseModel

# --- Modelo igual ao do treinamento ---
class TestAudioModel20s(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(320000, 3)

    def forward(self, x):
        return self.fc(x)

# --- Carregar modelo treinado ---
model = TestAudioModel20s()
model.load_state_dict(torch.load("model_20s.pt", map_location="cpu"))
model.eval()

app = FastAPI()

class PredictRequest(BaseModel):
    audio_base64: str

@app.post("/prever")
def predict(req: PredictRequest):
    # Converter base64 para numpy
    audio_bytes = base64.b64decode(req.audio_base64)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Garantir 20s = 320000 amostras
    TARGET = 320000
    if len(audio_np) > TARGET:
        audio_np = audio_np[:TARGET]
    elif len(audio_np) < TARGET:
        pad = TARGET - len(audio_np)
        audio_np = np.pad(audio_np, (0, pad), mode="constant")

    tensor = torch.from_numpy(audio_np).unsqueeze(0)  # shape [1, 320000]

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    labels = ["Normal", "Bronquite", "Pneumonia"]

    # Retornar todas as classes com percentual
    result = {labels[i]: float(probs[i]*100) for i in range(len(labels))}
    return result
