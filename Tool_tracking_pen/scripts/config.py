from pathlib import Path

# Projektordner automatisch bestimmen
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ordner für die ursprünglichen Datensätze
DATASETS_DIR = PROJECT_ROOT / "datasets"
PEN_RAW = DATASETS_DIR / "PenRaw"

# Rohbilder und zugehörige Masken
RAW_IMAGES = PEN_RAW / "images"
RAW_MASKS = PEN_RAW / "masks"

# nnU-Net Arbeitsordner
NNUNET_RAW = PROJECT_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = PROJECT_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = PROJECT_ROOT / "nnUNet_results"

# Ordner für die Vorhersage
PREDICTIONS_INPUT = PROJECT_ROOT / "predictions" / "input"
PREDICTIONS_OUTPUT = PROJECT_ROOT / "predictions" / "output"

# Dataset-ID und Name für nnU-Net
DATASET_ID = 100
DATASET_NAME = "Endo_own"

# Vollständiger nnU-Net Dataset-Ordner
NNUNET_DATASET = NNUNET_RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"