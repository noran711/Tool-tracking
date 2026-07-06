from pathlib import Path
import cv2
import numpy as np


# Pfade festlegen
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "videos" / "test7.mp4"
MASK_DIR = PROJECT_ROOT / "videos" / "predicted_masks"
OUTPUT_VIDEO = PROJECT_ROOT / "videos" / "tip_tracking8.mp4"


# Tracking-Einstellungen
MIN_AREA = 200
MAX_AREA = 20000
MIN_ASPECT_RATIO = 1.0

SEARCH_RADIUS = 200
LOST_REINIT_FRAMES = 1

USE_START_ROI = False
START_ROI_X_MIN = 0.35
START_ROI_X_MAX = 1.00
START_ROI_Y_MIN = 0.00
START_ROI_Y_MAX = 0.85

DRAW_MASK_OVERLAY = False
MASK_COLOR = (0, 255, 0)
MASK_ALPHA = 0.35

DRAW_TRAJECTORY = True
DRAW_TIP = True


# Kalman-Filter erstellen
def create_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)

    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman


# Suchbereich um die letzte Spitze anwenden
def apply_search_roi(binary_mask, last_tip, radius):
    if last_tip is None:
        return binary_mask

    h, w = binary_mask.shape
    x, y = last_tip

    x1 = max(0, x - radius)
    x2 = min(w, x + radius)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius)

    roi_mask = np.zeros_like(binary_mask)
    roi_mask[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]

    return roi_mask


# Maske binarisieren
def clean_mask(mask):
    return (mask > 0).astype(np.uint8) * 255


# Passende Kontur suchen
def get_pen_like_contour(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    valid_contours = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        rect = cv2.minAreaRect(c)
        rect_w, rect_h = rect[1]

        if rect_w == 0 or rect_h == 0:
            continue

        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)

        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        valid_contours.append(c)

    if len(valid_contours) == 0:
        return None

    return max(valid_contours, key=cv2.contourArea)


# Spitze aus der Maske bestimmen
def detect_tip_from_mask(binary_mask, last_tip=None):
    largest = get_pen_like_contour(binary_mask)

    if largest is None:
        return None

    # Hauptachse der Kontur berechnen
    vx, vy, x0, y0 = cv2.fitLine(
        largest,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    vx = vx.item()
    vy = vy.item()
    x0 = x0.item()
    y0 = y0.item()

    axis = np.array([vx, vy], dtype=np.float32)

    norm_axis = np.linalg.norm(axis)

    if norm_axis == 0:
        return None

    axis = axis / norm_axis

    # Richtung quer zur Hauptachse
    normal = np.array(
        [-axis[1], axis[0]],
        dtype=np.float32
    )

    ys, xs = np.where(binary_mask > 0)

    if len(xs) < 20:
        return None

    mask_points = np.column_stack((xs, ys)).astype(np.float32)

    # Punkte auf die Hauptachse projizieren
    projections = (
        (mask_points[:, 0] - x0) * axis[0] +
        (mask_points[:, 1] - y0) * axis[1]
    )

    proj_min = np.min(projections)
    proj_max = np.max(projections)
    length = proj_max - proj_min

    if length <= 10:
        return None

    # Endbereiche der Maske auswählen
    search_depth = max(
        10,
        int(0.08 * length)
    )

    selected_min = projections < proj_min + search_depth
    selected_max = projections > proj_max - search_depth

    region_min = mask_points[selected_min]
    region_max = mask_points[selected_max]

    if len(region_min) < 5 or len(region_max) < 5:
        return None

    # Breite eines Endbereichs berechnen
    def region_width(region):
        p_normal = (
            region[:, 0] * normal[0] +
            region[:, 1] * normal[1]
        )

        return np.max(p_normal) - np.min(p_normal)

    width_min = region_width(region_min)
    width_max = region_width(region_max)

    tip_min = np.mean(region_min, axis=0)
    tip_max = np.mean(region_max, axis=0)

    # Bei bekanntem Punkt das nähere Ende weiterverfolgen
    if last_tip is not None:
        last = np.array(last_tip, dtype=np.float32)

        d_min = np.linalg.norm(tip_min - last)
        d_max = np.linalg.norm(tip_max - last)

        if d_min < d_max:
            return tip_min.astype(int)

        return tip_max.astype(int)

    # Beim Start das schmalere Ende wählen
    if width_min < width_max:
        return tip_min.astype(int)

    return tip_max.astype(int)


# Maske transparent einzeichnen
def draw_mask_overlay(frame, binary_mask):
    result = frame.copy()
    overlay = frame.copy()

    overlay[binary_mask > 0] = MASK_COLOR

    blended = cv2.addWeighted(
        frame,
        1 - MASK_ALPHA,
        overlay,
        MASK_ALPHA,
        0
    )

    result[binary_mask > 0] = blended[binary_mask > 0]

    return result


# Video öffnen
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise Exception(f"Video konnte nicht geöffnet werden: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ausgabevideo vorbereiten
out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not out.isOpened():
    raise Exception(f"Ausgabevideo konnte nicht erstellt werden: {OUTPUT_VIDEO}")


# Tracking-Variablen
trajectory = []
last_tip = None
current_tip = None
lost_counter = 0
frame_idx = 0

kalman = create_kalman_filter()
kalman_initialized = False


# Erste gültige Spitze suchen
first_tip = None
init_frame_idx = 0

while first_tip is None:
    mask_path = MASK_DIR / f"frame_{init_frame_idx:04d}.png"

    if not mask_path.exists():
        init_frame_idx += 1

        if init_frame_idx > 10000:
            raise Exception("Keine gültige Maske zur Initialisierung gefunden.")

        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        init_frame_idx += 1
        continue

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    binary = clean_mask(mask)

    candidate_tip = detect_tip_from_mask(
        binary,
        last_tip=None
    )

    if candidate_tip is not None:
        x_tip = int(candidate_tip[0])
        y_tip = int(candidate_tip[1])

        if USE_START_ROI:
            inside_roi = (
                START_ROI_X_MIN * width <= x_tip <= START_ROI_X_MAX * width and
                START_ROI_Y_MIN * height <= y_tip <= START_ROI_Y_MAX * height
            )

            if inside_roi:
                first_tip = candidate_tip

        else:
            first_tip = candidate_tip

    init_frame_idx += 1


# Kalman-Filter mit erster Spitze starten
x, y = int(first_tip[0]), int(first_tip[1])

initial_state = np.array(
    [
        [np.float32(x)],
        [np.float32(y)],
        [0.0],
        [0.0]
    ],
    dtype=np.float32
)

kalman.statePost = initial_state.copy()
kalman.statePre = initial_state.copy()

kalman_initialized = True
last_tip = (x, y)
trajectory.append(last_tip)

print(f"Kalman initialisiert mit erster Masken-Spitze: {last_tip}")


# Video Frame für Frame verarbeiten
while True:
    ret, frame = cap.read()

    if not ret:
        break

    mask_path = MASK_DIR / f"frame_{frame_idx:04d}.png"
    current_tip = None
    binary = None

    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

            binary = clean_mask(mask)

            # Bei stabilem Tracking nur in der Nähe suchen
            if last_tip is not None and lost_counter < LOST_REINIT_FRAMES:
                search_mask = apply_search_roi(
                    binary,
                    last_tip,
                    SEARCH_RADIUS
                )

            else:
                search_mask = binary

            raw_tip = detect_tip_from_mask(
                search_mask,
                last_tip=last_tip
            )

            if raw_tip is not None:
                raw_tip_tuple = (
                    int(raw_tip[0]),
                    int(raw_tip[1])
                )

                measurement = np.array(
                    [
                        [np.float32(raw_tip_tuple[0])],
                        [np.float32(raw_tip_tuple[1])]
                    ],
                    dtype=np.float32
                )

                # Messung mit Kalman glätten
                kalman.predict()
                kalman.correct(measurement)

                kalman_tip = (
                    int(kalman.statePost[0].item()),
                    int(kalman.statePost[1].item())
                )

                trajectory.append(kalman_tip)
                last_tip = kalman_tip
                current_tip = kalman_tip
                lost_counter = 0

            else:
                lost_counter += 1

                # Kurzzeitig fehlende Spitze vorhersagen
                if kalman_initialized and lost_counter <= LOST_REINIT_FRAMES:
                    prediction = kalman.predict()

                    kalman_tip = (
                        int(prediction[0].item()),
                        int(prediction[1].item())
                    )

                    trajectory.append(kalman_tip)
                    last_tip = kalman_tip
                    current_tip = kalman_tip

    # Frame für Anzeige vorbereiten
    display_frame = frame.copy()

    if DRAW_MASK_OVERLAY and binary is not None:
        display_frame = draw_mask_overlay(
            display_frame,
            binary
        )

    # Trajektorie einzeichnen
    if DRAW_TRAJECTORY:
        for i in range(1, len(trajectory)):
            cv2.line(
                display_frame,
                trajectory[i - 1],
                trajectory[i],
                (0, 255, 0),
                2
            )

    # Aktuelle Spitze einzeichnen
    if DRAW_TIP and current_tip is not None:
        cv2.circle(
            display_frame,
            current_tip,
            8,
            (0, 0, 255),
            -1
        )

    out.write(display_frame)

    cv2.imshow(
        "Tip Tracking",
        display_frame
    )

    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    if key == 27:
        break

    frame_idx += 1


# Ressourcen freigeben
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Visualisiertes Video gespeichert unter: {OUTPUT_VIDEO}")