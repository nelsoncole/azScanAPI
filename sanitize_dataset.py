import os
import soundfile as sf

ROOT = "dataset"

removidos = 0
total = 0

for cls in os.listdir(ROOT):
    cls_path = os.path.join(ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    for f in os.listdir(cls_path):
        if not f.lower().endswith(".wav"):
            continue

        total += 1
        path = os.path.join(cls_path, f)

        try:
            sf.read(path)
        except Exception:
            print(f"Removendo WAV inválido: {path}")
            os.remove(path)
            removidos += 1

print(f"\nLimpeza concluída")
print(f"Total analisados: {total}")
print(f"Removidos: {removidos}")
