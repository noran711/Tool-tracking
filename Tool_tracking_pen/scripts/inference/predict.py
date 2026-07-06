from pathlib import Path
import sys
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# config.py einbinden
PROJECT_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SCRIPTS))
import config as cfg


# nnU-Net Pfade setzen
os.environ["nnUNet_raw"] = str(cfg.NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(cfg.NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(cfg.NNUNET_RESULTS)


# Input- und Outputordner festlegen
INPUT_DIR = cfg.PREDICTIONS_INPUT
OUTPUT_DIR = cfg.PREDICTIONS_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Temporärer Ordner für einzelne Vorhersagen
TEMP_INPUT = PROJECT_SCRIPTS / "temp_input"
TEMP_INPUT.mkdir(exist_ok=True)

# Pfad zum trainierten Modell
MODEL_FOLDER = (
    cfg.NNUNET_RESULTS
    / f"Dataset{cfg.DATASET_ID:03d}_{cfg.DATASET_NAME}"
    / f"nnUNetTrainer_100epochs__nnUNetPlans__2d"
)


# Predictor vorbereiten
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=False,
    device=torch.device("cpu"),
    verbose=False,
    allow_tqdm=True,
)

# Trainiertes Modell laden
predictor.initialize_from_trained_model_folder(
    str(MODEL_FOLDER),
    use_folds=(0,),
    checkpoint_name="checkpoint_final.pth",
)


# Alle Input-Bilder sammeln
input_files = sorted(
    list(INPUT_DIR.glob("*.png")) +
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg"))
)

if len(input_files) == 0:
    raise Exception("Keine Input-Bilder gefunden.")


# Vorhersage für jedes Bild ausführen
for img_path in input_files:
    print(f"Predicte {img_path.name} ...")

    # Temp-Ordner leeren
    for f in TEMP_INPUT.iterdir():
        f.unlink()

    # Bild in den Temp-Ordner kopieren
    shutil.copy(str(img_path), TEMP_INPUT)

    # Eigener Output-Ordner für jedes Bild
    single_output = OUTPUT_DIR / img_path.stem
    single_output.mkdir(exist_ok=True)

    # Prediction starten
    predictor.predict_from_files(
        str(TEMP_INPUT),
        str(single_output),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
    )

    print(f"Prediction abgeschlossen: {img_path.name}")


# Ergebnisbilder anzeigen
for img_path in input_files:
    img_name = img_path.stem
    mask_folder = OUTPUT_DIR / img_name

    # Prediction-Maske suchen
    mask_files = list(mask_folder.glob("*.png"))
    if len(mask_files) == 0:
        print(f"⚠ Keine Prediction-Maske für {img_path.name} gefunden.")
        continue

    mask_path = mask_files[0]

    # Bild und Maske laden
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"⚠ Fehler bei {img_path.name}")
        continue

    # Maske an Bildgröße anpassen
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Overlay erstellen
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("Prediction Mask")
    plt.imshow(mask, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, cmap="jet", alpha=0.5)

    plt.tight_layout()
    plt.show()

    print(f"Overlay fertig: {img_path.name}")