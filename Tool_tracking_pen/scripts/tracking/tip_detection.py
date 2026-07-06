from pathlib import Path
import sys
import cv2
import numpy as np

# config.py einbinden
PROJECT_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SCRIPTS))
import config as cfg


# Input- und Outputordner
INPUT_DIR = cfg.PREDICTIONS_INPUT
OUTPUT_DIR = cfg.PREDICTIONS_OUTPUT
OVERLAY_DIR = OUTPUT_DIR / "overlays"

OVERLAY_DIR.mkdir(exist_ok=True)


# Darstellungseinstellungen
MASK_COLOR = (0, 255, 0)   # Grün, BGR-Format
ALPHA = 0.4                # Transparenz der Maske
TIP_COLOR = (0, 0, 255)    # Rot, BGR-Format
TIP_RADIUS = 8


# Spitze aus der Maske bestimmen
def detect_tip_from_mask(binary):
    ys, xs = np.where(binary > 0)

    if len(xs) < 20:
        return None

    # Maskenpunkte sammeln
    points = np.column_stack((xs, ys)).astype(np.float32)

    # Schwerpunkt berechnen
    mean = np.mean(points, axis=0)

    # Punkte zentrieren
    centered = points - mean

    # Hauptachse der Maske berechnen
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    axis = eigenvectors[:, np.argmax(eigenvalues)]
    axis = axis / np.linalg.norm(axis)

    # Richtung quer zur Hauptachse
    normal = np.array(
        [-axis[1], axis[0]],
        dtype=np.float32
    )

    # Punkte auf die Hauptachse projizieren
    proj = centered @ axis

    proj_min = np.min(proj)
    proj_max = np.max(proj)

    length = proj_max - proj_min

    if length < 10:
        return None

    # Endbereiche der Maske auswählen
    search_depth = max(
        10,
        int(0.08 * length)
    )

    region_min = points[
        proj < proj_min + search_depth
    ]

    region_max = points[
        proj > proj_max - search_depth
    ]

    # Breite eines Endbereichs berechnen
    def region_width(region):
        if len(region) < 5:
            return 9999

        p_norm = (region - mean) @ normal

        return np.max(p_norm) - np.min(p_norm)

    width_min = region_width(region_min)
    width_max = region_width(region_max)

    # Endpunkte berechnen
    end_min = np.mean(region_min, axis=0)
    end_max = np.mean(region_max, axis=0)

    # Schmaleres Ende als Spitze wählen
    if width_min < width_max:
        tip = end_min
    else:
        tip = end_max

    return tip.astype(int)


# Alle Input-Bilder sammeln
input_files = sorted(
    list(INPUT_DIR.glob("*.png")) +
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg"))
)


for img_path in input_files:

    img_name = img_path.stem

    # Passende Maske suchen
    mask_path = None

    mask_folder = OUTPUT_DIR / img_name

    if mask_folder.exists():

        mask_files = list(
            mask_folder.glob("*.png")
        )

        if mask_files:
            mask_path = mask_files[0]

    if mask_path is None:

        mask_files = list(
            OUTPUT_DIR.glob("*.png")
        )

        if mask_files:
            mask_path = mask_files[0]

        else:
            print(
                f"⚠ Keine Maske gefunden für {img_name}"
            )
            continue


    # Bild und Maske laden
    img = cv2.imread(
        str(img_path)
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None or mask is None:
        print(
            f"⚠ Fehler bei {img_name}"
        )
        continue


    # Maske an Bildgröße anpassen
    mask = cv2.resize(
        mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Binärmaske erzeugen
    binary = (
        (mask > 0)
        .astype(np.uint8)
        * 255
    )


    # Spitze berechnen
    tip = detect_tip_from_mask(binary)


    # Overlay vorbereiten
    result = img.copy()
    overlay = img.copy()

    # Maske einfärben
    overlay[binary > 0] = MASK_COLOR

    # Maske transparent überlagern
    blended = cv2.addWeighted(
        img,
        1 - ALPHA,
        overlay,
        ALPHA,
        0
    )

    # Overlay nur im Maskenbereich übernehmen
    result[binary > 0] = blended[binary > 0]


    # Spitze einzeichnen
    if tip is not None:

        cv2.circle(
            result,
            tuple(tip),
            TIP_RADIUS,
            TIP_COLOR,
            -1
        )

        print(
            f"Spitze für {img_name}: {tip}"
        )

    else:

        print(
            f"⚠ Keine Spitze gefunden für {img_name}"
        )


    # Ergebnis speichern
    overlay_path = (
        OVERLAY_DIR
        /
        f"{img_name}_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        result
    )


print("\nFertig")