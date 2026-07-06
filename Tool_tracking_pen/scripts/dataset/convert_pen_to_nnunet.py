from pathlib import Path
import sys
import json
import cv2
import numpy as np

# config.py aus dem übergeordneten Ordner einbinden
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg


# Zielordner für nnU-Net anlegen
imagesTr = cfg.NNUNET_DATASET / "imagesTr"
labelsTr = cfg.NNUNET_DATASET / "labelsTr"
imagesTs = cfg.NNUNET_DATASET / "imagesTs"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)
imagesTs.mkdir(parents=True, exist_ok=True)


# Alle Bilddateien aus dem Rohdatenordner sammeln
image_files = sorted(
    list(cfg.RAW_IMAGES.glob("*.jpg")) +
    list(cfg.RAW_IMAGES.glob("*.jpeg")) +
    list(cfg.RAW_IMAGES.glob("*.png"))
)

converted = 0

for img_path in image_files:
    stem = img_path.stem

    # Passende Maske zum Bild suchen
    mask_path = cfg.RAW_MASKS / f"{stem}.png"

    if not mask_path.exists():
        print(f"Keine Maske gefunden für: {img_path.name}")
        continue

    # Bild und Maske als Graustufen laden
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Bild konnte nicht gelesen werden: {img_path}")
        continue

    if mask is None:
        print(f"Maske konnte nicht gelesen werden: {mask_path}")
        continue

    # Bildgröße an die Maskengröße anpassen
    img = cv2.resize(
        img,
        (mask.shape[1], mask.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    # Maske in Hintergrund und Stift aufteilen
    label = (mask > 0).astype(np.uint8)

    case_name = f"stift_{converted:04d}"

    # Bild im nnU-Net Format speichern
    cv2.imwrite(
        str(imagesTr / f"{case_name}_0000.png"),
        img
    )

    # Maske im nnU-Net Format speichern
    cv2.imwrite(
        str(labelsTr / f"{case_name}.png"),
        label
    )

    converted += 1


# Beschreibung des Datensatzes für nnU-Net
dataset_json = {
    "channel_names": {
        "0": "Grayscale"
    },
    "labels": {
        "background": 0,
        "pen": 1
    },
    "numTraining": converted,
    "file_ending": ".png"
}

# dataset.json speichern
with open(cfg.NNUNET_DATASET / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Fertig. {converted} Bilder konvertiert.")
print(f"Gespeichert unter: {cfg.NNUNET_DATASET}")