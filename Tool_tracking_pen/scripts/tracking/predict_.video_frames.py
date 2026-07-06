from pathlib import Path
import sys
import os
import subprocess
import cv2

# config.py laden
PROJECT_SCRIPTS = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_SCRIPTS))
import config as cfg


# nnU-Net Pfade setzen
os.environ["nnUNet_raw"] = str(cfg.NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(cfg.NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(cfg.NNUNET_RESULTS)


# Video- und Ausgabeordner
VIDEO_PATH = PROJECT_ROOT / "videos" / "test8.mp4"

FRAME_DIR = PROJECT_ROOT / "videos" / "frames_for_prediction"
MASK_DIR = PROJECT_ROOT / "videos" / "predicted_masks"


# nnU-Net Einstellungen
DATASET_ID = "100"
CONFIGURATION = "2d"
FOLD = "0"
TRAINER = "nnUNetTrainer"
CHECKPOINT = "checkpoint_final.pth"
DEVICE = "cpu"

TARGET_SIZE = (432, 432)


# Ordner erstellen und alte Dateien löschen
FRAME_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

for folder in [FRAME_DIR, MASK_DIR]:
    for f in folder.glob("*"):
        if f.is_file():
            f.unlink()


# Video öffnen
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise Exception(f"Video konnte nicht geöffnet werden: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video geladen: {VIDEO_PATH}")
print(f"FPS: {fps}, Größe: {width}x{height}")


# Video Frame für Frame speichern
frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Frame auf Modellgröße bringen
    resized = cv2.resize(
        frame,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Farbkanäle trennen
    b, g, r = cv2.split(resized)

    case_name = f"frame_{frame_idx:04d}"

    # Kanäle im nnU-Net Format speichern
    cv2.imwrite(str(FRAME_DIR / f"{case_name}_0000.png"), b)
    cv2.imwrite(str(FRAME_DIR / f"{case_name}_0001.png"), g)
    cv2.imwrite(str(FRAME_DIR / f"{case_name}_0002.png"), r)

    frame_idx += 1

cap.release()

print(f"{frame_idx} Frames gespeichert in: {FRAME_DIR}")
print(f"{frame_idx * 3} Kanal-Dateien gespeichert.")


# Beispiel-Frame prüfen
example_files = sorted(FRAME_DIR.glob("frame_0000_*.png"))

print("Beispiel-Dateien für frame_0000:")
for f in example_files:
    print(f.name)

if len(example_files) != 3:
    raise RuntimeError(
        "Fehler: Für frame_0000 wurden nicht 3 Kanal-Dateien gespeichert. "
        "Erwartet: frame_0000_0000.png, frame_0000_0001.png, frame_0000_0002.png"
    )


# Anzahl der gespeicherten Dateien prüfen
all_pngs = list(FRAME_DIR.glob("*.png"))
expected_pngs = frame_idx * 3

print(f"PNG-Dateien im Input-Ordner: {len(all_pngs)}")
print(f"Erwartete PNG-Dateien: {expected_pngs}")

if len(all_pngs) != expected_pngs:
    raise RuntimeError(
        f"Fehler: Es sollten {expected_pngs} PNG-Dateien vorhanden sein, "
        f"gefunden wurden aber {len(all_pngs)}."
    )


# nnU-Net Befehl vorbereiten
cmd = [
    "nnUNetv2_predict",
    "-i", str(FRAME_DIR),
    "-o", str(MASK_DIR),
    "-d", DATASET_ID,
    "-c", CONFIGURATION,
    "-f", FOLD,
    "-tr", TRAINER,
    "-chk", CHECKPOINT,
    "-device", DEVICE
]


# Prediction starten
print("Starte nnU-Net Prediction...")
print(" ".join(cmd))

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True
)


# Ausgabe anzeigen
print("STDOUT:")
print(result.stdout)

print("STDERR:")
print(result.stderr)

result.check_returncode()

print(f"Prediction abgeschlossen. Masken gespeichert in: {MASK_DIR}")