# pip install torch numpy soundfile torchaudio
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio  # só para resample se necessário

# --- 1️ Dataset personalizado
class CoughDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for f in os.listdir(cls_path):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(cls_path, f))
                    self.labels.append(i)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Ler WAV com soundfile
        wav_np, sr = sf.read(self.files[idx])
        print(f"Lendo arquivo: {self.files[idx]}, shape: {wav_np.shape}, sr: {sr}")
        wav = torch.from_numpy(wav_np).unsqueeze(0).float()

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample para 16kHz se necessário
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # Garantir 20s = 320000 amostras
        TARGET = 320000
        if wav.shape[1] > TARGET:
            wav = wav[:, :TARGET]
        elif wav.shape[1] < TARGET:
            pad = TARGET - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))

        return wav.squeeze(0), self.labels[idx]

# --- 2️ Modelo ---
class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(320000, 3)  # 3 classes

    def forward(self, x):
        return self.fc(x)

# --- 3️ Treinamento ---
def train_model(data_dir, epochs=5, batch_size=2, lr=0.001):
    dataset = CoughDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioModel()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Época {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

    # Salvar modelo treinado
    torch.save(model.state_dict(), "model_20s.pt")
    print("Modelo treinado salvo como model_20s.pt")

# --- 4️ Executar ---
if __name__ == "__main__":
    data_dir = "dataset"  # pasta com normal/, bronquite/, pneumonia/
    train_model(data_dir, epochs=5, batch_size=2, lr=0.001)
