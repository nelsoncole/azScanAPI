# pip install torch numpy soundfile torchaudio scikit-learn matplotlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import soundfile as sf
import torchaudio

# ---- métricas ----
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ===================== DATASET =====================
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
        wav_np, sr = sf.read(self.files[idx])
        print(f"Lendo arquivo: {self.files[idx]}, shape: {wav_np.shape}, sr: {sr}")
        wav = torch.from_numpy(wav_np).unsqueeze(0).float()

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        TARGET = 320000
        if wav.shape[1] > TARGET:
            wav = wav[:, :TARGET]
        else:
            wav = torch.nn.functional.pad(wav, (0, TARGET - wav.shape[1]))

        return wav.squeeze(0), self.labels[idx]

# ===================== MODELO =====================
class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(320000, 3)

    def forward(self, x):
        return self.fc(x)

# ===================== TREINO =====================
def train_model(data_dir, epochs=5, batch_size=2, lr=0.001):
    dataset = CoughDataset(data_dir)

    # ---- split treino / teste (obrigatório para métricas) ----
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AudioModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Época {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "model_20s.pt")
    print("Modelo treinado salvo como model_20s.pt")

    avaliar_modelo(model, test_loader, dataset.classes)

# ===================== AVALIAÇÃO =====================
def avaliar_modelo(model, dataloader, classes):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).numpy())
            y_prob.extend(probs.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # -------- ACURÁCIA --------
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia do modelo: {acc:.4f}")

    # -------- MATRIZ DE CONFUSÃO --------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.tight_layout()
    plt.savefig("matriz_confusao.png")
    plt.close()

    # -------- CURVA ROC --------
    y_bin = label_binarize(y_true, classes=range(len(classes)))

    plt.figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Falso Positive")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Curva ROC")
    plt.legend()
    plt.savefig("curva_roc.png")
    plt.close()

    print("Matriz de confusão salva: matriz_confusao.png")
    print("Curva ROC salva: curva_roc.png")

# ===================== EXECUÇÃO =====================
if __name__ == "__main__":
    train_model("dataset", epochs=5, batch_size=2, lr=0.001)
