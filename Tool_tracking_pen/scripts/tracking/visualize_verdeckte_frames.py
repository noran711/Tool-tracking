from pathlib import Path
import cv2
import numpy as np


# Pfade festlegen
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "videos" / "test7.mp4"
MASK_DIR = PROJECT_ROOT / "videos" / "verdeckt2" / "predicted_masks"
OUTPUT_VIDEO = PROJECT_ROOT / "videos" / "rightmost_tip_edge_normal_shorter_prediction.mp4"


# Grundeinstellungen
MIN_AREA = 200
MAX_AREA = 50000

LEARN_FROM_FIRST_N_FRAMES = 80

RIGHT_TIP_PERCENTILE = 99.5
LEFT_BASE_PERCENTILE = 0.5

SHAFT_MIDDLE_START = 0.25
SHAFT_MIDDLE_END = 0.75

OCCLUDED_AXIS_START = 0.45
OCCLUDED_AXIS_END = 0.90

LENGTH_UPDATE_ALPHA = 0.04

OCCLUSION_LENGTH_RATIO = 0.82
RECOVERY_LENGTH_RATIO = 0.92

# Korrekturfaktor für die geschätzte Spitze
PREDICTION_LENGTH_SCALE = 0.90

TIP_BASE_START = 0.70
TIP_BASE_END = 0.86
TIP_END_START = 0.88
TIP_END_END = 0.99
TIP_OCCLUDED_TAPER_RATIO = 0.85
TIP_VISIBLE_TAPER_RATIO = 0.65

MAX_AXIS_ANGLE_CHANGE_DEG = 25
MAX_TIP_JUMP = 180
MAX_LOST_FRAMES = 60

SHOW_VIDEO = True
SAVE_VIDEO = True


# Passende Maske zum Frame suchen
def find_mask_path(frame_idx):
    names = [
        f"frame_{frame_idx:04d}.png",
        f"{frame_idx:04d}.png",
        f"frame_{frame_idx}.png",
        f"{frame_idx}.png",
    ]

    for name in names:
        path = MASK_DIR / name
        if path.exists():
            return path

    return None


# Maske laden und bereinigen
def load_mask(mask_path, width, height):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# Größte gültige Komponente wählen
def get_largest_component(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    best_label = None
    best_area = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if MIN_AREA <= area <= MAX_AREA and area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return None

    return (labels == best_label).astype(np.uint8)


# Hauptachse mit PCA berechnen
def compute_pca_axis(points):
    center = np.mean(points, axis=0).astype(np.float32)
    centered = points - center

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    axis = eigenvectors[:, np.argmax(eigenvalues)].astype(np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-6)

    return center, axis


# Winkel zwischen zwei Achsen berechnen
def angle_between_axes(axis_a, axis_b):
    dot = float(np.clip(np.dot(axis_a, axis_b), -1.0, 1.0))
    return np.degrees(np.arccos(abs(dot)))


# Achse aus der vorderen Kante bestimmen
def axis_from_front_edge_normal(points, prefer_right=True):
    if points is None or len(points) < 8:
        return None, None, None

    xs = points[:, 0]

    if prefer_right:
        edge_threshold = np.percentile(xs, 97)
        edge_points = points[xs >= edge_threshold]
    else:
        edge_threshold = np.percentile(xs, 3)
        edge_points = points[xs <= edge_threshold]

    if len(edge_points) < 5:
        return None, None, None

    vx, vy, x0, y0 = cv2.fitLine(
        edge_points.astype(np.float32),
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    edge_dir = np.array([vx.item(), vy.item()], dtype=np.float32)
    edge_dir = edge_dir / (np.linalg.norm(edge_dir) + 1e-6)

    normal_1 = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float32)
    normal_2 = -normal_1

    axis = normal_1 if normal_1[0] >= normal_2[0] else normal_2
    axis = axis / (np.linalg.norm(axis) + 1e-6)

    center = np.mean(edge_points, axis=0).astype(np.float32)

    return center, axis, edge_points


# Geometrie aus der Maske extrahieren
def extract_geometry(
    binary,
    last_axis=None,
    axis_start=SHAFT_MIDDLE_START,
    axis_end=SHAFT_MIDDLE_END,
    use_edge_normal=False
):
    component = get_largest_component(binary)

    if component is None:
        return None

    ys, xs = np.where(component > 0)

    if len(xs) < 20:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)

    rough_center, rough_axis = compute_pca_axis(points)

    if last_axis is not None and np.dot(rough_axis, last_axis) < 0:
        rough_axis = -rough_axis

    rough_proj = np.dot(points - rough_center, rough_axis)

    q_low = np.percentile(rough_proj, 0.5)
    q_high = np.percentile(rough_proj, 99.5)
    visible_length = float(q_high - q_low)

    if visible_length < 10:
        return None

    region_low = q_low + axis_start * visible_length
    region_high = q_low + axis_end * visible_length

    region_mask = (
        (rough_proj >= region_low)
        & (rough_proj <= region_high)
    )

    region_points = points[region_mask]

    edge_points = None

    if use_edge_normal and len(region_points) >= 8:
        edge_center, edge_axis, edge_points = axis_from_front_edge_normal(
            region_points,
            prefer_right=True
        )

        if edge_axis is not None:
            center = edge_center
            middle_axis = edge_axis
        else:
            center, middle_axis = compute_pca_axis(region_points)
    else:
        if len(region_points) >= 20:
            center, middle_axis = compute_pca_axis(region_points)
        else:
            center = rough_center
            middle_axis = rough_axis
            region_points = points

    if last_axis is not None:
        if np.dot(middle_axis, last_axis) < 0:
            middle_axis = -middle_axis

        angle_change = angle_between_axes(middle_axis, last_axis)

        max_angle = MAX_AXIS_ANGLE_CHANGE_DEG if not use_edge_normal else 45

        if angle_change > max_angle:
            middle_axis = last_axis.copy()

    if middle_axis[0] < 0:
        middle_axis = -middle_axis

    proj = np.dot(points - center, middle_axis)

    p_low = np.percentile(proj, 0.5)
    p_high = np.percentile(proj, 99.5)

    left_end = center + middle_axis * p_low
    right_end = center + middle_axis * p_high

    return {
        "component": component,
        "points": points,
        "middle_points": region_points,
        "edge_points": edge_points,
        "center": center,
        "middle_axis": middle_axis,
        "left_end": left_end,
        "right_end": right_end,
        "visible_length": float(p_high - p_low),
        "axis_start": axis_start,
        "axis_end": axis_end,
        "use_edge_normal": use_edge_normal,
    }


# Rechte Spitze und linke Basis bestimmen
def choose_rightmost_tip_and_left_base(geometry):
    points = geometry["points"]
    xs = points[:, 0]

    tip_x_threshold = np.percentile(xs, RIGHT_TIP_PERCENTILE)
    base_x_threshold = np.percentile(xs, LEFT_BASE_PERCENTILE)

    tip_points = points[xs >= tip_x_threshold]
    base_points = points[xs <= base_x_threshold]

    if len(tip_points) < 3:
        tip = points[np.argmax(xs)]
    else:
        tip = np.mean(tip_points, axis=0)

    if len(base_points) < 3:
        base = points[np.argmin(xs)]
    else:
        base = np.mean(base_points, axis=0)

    return tip.astype(np.float32), base.astype(np.float32)


# Richtung von Basis zu Spitze berechnen
def axis_from_base_to_tip(base, tip):
    direction = tip - base
    norm = np.linalg.norm(direction)

    if norm < 1e-6:
        return None

    return direction / norm


# Breite in einem Abschnitt berechnen
def section_width(points, base, direction, start_ratio, end_ratio):
    projections = np.dot(points - base, direction)
    max_projection = np.max(projections)

    start = max_projection * start_ratio
    end = max_projection * end_ratio

    section = points[
        (projections >= start)
        & (projections <= end)
    ]

    if len(section) < 5:
        return None

    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    side_values = np.dot(section - base, perpendicular)

    width = np.percentile(side_values, 90) - np.percentile(side_values, 10)

    return float(width)


# Verjüngung der Spitze berechnen
def tip_taper_ratio(geometry, base, tip):
    direction = axis_from_base_to_tip(base, tip)

    if direction is None:
        return None

    base_width = section_width(
        geometry["points"],
        base,
        direction,
        TIP_BASE_START,
        TIP_BASE_END
    )

    end_width = section_width(
        geometry["points"],
        base,
        direction,
        TIP_END_START,
        TIP_END_END
    )

    if base_width is None or end_width is None:
        return None

    return end_width / max(base_width, 1e-6)


# Kalman-Filter erstellen
def create_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)

    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman


# Kalman-Filter initialisieren
def initialize_kalman(kalman, point):
    state = np.array([
        [np.float32(point[0])],
        [np.float32(point[1])],
        [0.0],
        [0.0],
    ], dtype=np.float32)

    kalman.statePost = state.copy()
    kalman.statePre = state.copy()


# Kalman-Ausgabe als Punkt umwandeln
def kalman_point(value):
    return np.array([
        value[0].item(),
        value[1].item(),
    ], dtype=np.float32)


# Kalman-Filter mit Messung korrigieren
def correct_kalman(kalman, point):
    measurement = np.array([
        [np.float32(point[0])],
        [np.float32(point[1])],
    ], dtype=np.float32)

    corrected = kalman.correct(measurement)
    return kalman_point(corrected)


# Maske über das Bild legen
def overlay_mask(frame, binary, color=(255, 0, 255), alpha=0.30):
    result = frame.copy()

    color_layer = np.zeros_like(frame)
    color_layer[binary > 0] = color

    blended = cv2.addWeighted(frame, 1.0, color_layer, alpha, 0)

    mask_3ch = np.repeat((binary > 0)[:, :, None], 3, axis=2)
    result[mask_3ch] = blended[mask_3ch]

    return result


# Punkte einzeichnen
def draw_points(frame, points, color, max_points=800):
    if points is None or len(points) == 0:
        return

    step = max(1, len(points) // max_points)

    for point in points[::step]:
        cv2.circle(frame, (int(point[0]), int(point[1])), 1, color, -1)


# Text ins Bild schreiben
def draw_text(frame, text, y):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# Video öffnen
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise Exception(f"Video konnte nicht geoeffnet werden: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Referenzlänge aus den ersten Frames lernen
lengths = []

for frame_idx in range(LEARN_FROM_FIRST_N_FRAMES):
    mask_path = find_mask_path(frame_idx)

    if mask_path is None:
        continue

    binary = load_mask(mask_path, width, height)

    if binary is None:
        continue

    geometry = extract_geometry(
        binary,
        axis_start=SHAFT_MIDDLE_START,
        axis_end=SHAFT_MIDDLE_END,
        use_edge_normal=False
    )

    if geometry is not None:
        lengths.append(geometry["visible_length"])

if not lengths:
    raise Exception("Keine brauchbare Maske gefunden.")

lengths = sorted(lengths, reverse=True)
known_length = float(np.median(lengths[:max(3, len(lengths) // 5)]))

print(f"Gelernte Tool-Laenge: {known_length:.1f}px")


# Video wieder auf Anfang setzen
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

out = None

# Ausgabevideo vorbereiten
if SAVE_VIDEO:
    out = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

# Tracking-Variablen
kalman = create_kalman_filter()
kalman_initialized = False

last_axis = None
last_tip = None

occlusion_mode = False
trajectory = []

lost_counter = 0
frame_idx = 0


# Video Frame für Frame verarbeiten
while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = frame.copy()

    current_tip = None
    raw_tip = None
    base = None
    predicted_tip = None
    reconstructed_tip = None
    geometry = None
    taper_ratio = None
    binary = None

    used_prediction = False
    status = "lost"

    # Kalman-Vorhersage berechnen
    if kalman_initialized:
        prediction = kalman.predict()
        predicted_tip = kalman_point(prediction)

    mask_path = find_mask_path(frame_idx)

    # Maske für aktuellen Frame laden
    if mask_path is not None:
        binary = load_mask(mask_path, width, height)

        if binary is not None:
            geometry = extract_geometry(
                binary,
                last_axis,
                axis_start=SHAFT_MIDDLE_START,
                axis_end=SHAFT_MIDDLE_END,
                use_edge_normal=False
            )
            result = overlay_mask(result, binary)

    if geometry is not None:
        # Spitze und Basis bestimmen
        raw_tip, base = choose_rightmost_tip_and_left_base(geometry)
        rightmost_axis = axis_from_base_to_tip(base, raw_tip)

        middle_axis = geometry["middle_axis"].copy()

        if middle_axis[0] < 0:
            middle_axis = -middle_axis

        taper_ratio = tip_taper_ratio(geometry, base, raw_tip)

        # Prüfen, ob die Spitze verdeckt ist
        length_says_occluded = (
            geometry["visible_length"]
            < known_length * OCCLUSION_LENGTH_RATIO
        )

        shape_says_occluded = False
        shape_says_visible = False

        if taper_ratio is not None:
            shape_says_occluded = taper_ratio >= TIP_OCCLUDED_TAPER_RATIO
            shape_says_visible = taper_ratio <= TIP_VISIBLE_TAPER_RATIO

        if length_says_occluded or shape_says_occluded:
            occlusion_mode = True

        if occlusion_mode:
            if (
                geometry["visible_length"] > known_length * RECOVERY_LENGTH_RATIO
                and shape_says_visible
            ):
                occlusion_mode = False

        # Bei Verdeckung Achse aus Kante bestimmen
        if occlusion_mode and binary is not None:
            edge_geometry = extract_geometry(
                binary,
                last_axis,
                axis_start=OCCLUDED_AXIS_START,
                axis_end=OCCLUDED_AXIS_END,
                use_edge_normal=True
            )

            if edge_geometry is not None:
                geometry = edge_geometry
                raw_tip, base = choose_rightmost_tip_and_left_base(geometry)
                middle_axis = geometry["middle_axis"].copy()

                if middle_axis[0] < 0:
                    middle_axis = -middle_axis

        # Messpunkt wählen
        if occlusion_mode:
            reconstructed_tip = (
                base
                + middle_axis * known_length * PREDICTION_LENGTH_SCALE
            )
            measurement_tip = reconstructed_tip
            used_prediction = True
            status = "occluded_edge_normal_shorter_prediction"
            active_axis = middle_axis
        else:
            measurement_tip = raw_tip
            used_prediction = False
            status = "rightmost_tip_visible"
            active_axis = rightmost_axis if rightmost_axis is not None else middle_axis

            known_length = (
                known_length * (1.0 - LENGTH_UPDATE_ALPHA)
                + geometry["visible_length"] * LENGTH_UPDATE_ALPHA
            )

        # Kalman-Filter starten oder aktualisieren
        if not kalman_initialized:
            initialize_kalman(kalman, measurement_tip)
            kalman_initialized = True
            current_tip = measurement_tip
            lost_counter = 0
            status = "init_" + status

        else:
            jump_ok = True

            if predicted_tip is not None:
                jump = np.linalg.norm(measurement_tip - predicted_tip)
                jump_ok = jump <= MAX_TIP_JUMP

            if jump_ok:
                current_tip = correct_kalman(kalman, measurement_tip)
                lost_counter = 0
            else:
                lost_counter += 1

                if predicted_tip is not None and lost_counter <= MAX_LOST_FRAMES:
                    current_tip = predicted_tip
                    used_prediction = True
                    status = "kalman_prediction_jump_rejected"
                else:
                    current_tip = None
                    status = "lost_jump"

        last_axis = active_axis.copy()

    else:
        lost_counter += 1

        # Bei fehlender Maske kurz mit Kalman weiterlaufen
        if predicted_tip is not None and lost_counter <= MAX_LOST_FRAMES:
            current_tip = predicted_tip
            used_prediction = True
            status = "kalman_no_mask"
        else:
            current_tip = None
            status = "lost_no_mask"

    if current_tip is not None:
        tip_point = (
            int(round(current_tip[0])),
            int(round(current_tip[1]))
        )

        trajectory.append(tip_point)
        last_tip = current_tip.copy()

        color = (0, 165, 255) if used_prediction else (0, 0, 255)
        cv2.circle(result, tip_point, 9, color, -1)

    # Rohspitze einzeichnen
    if raw_tip is not None:
        cv2.circle(
            result,
            (int(round(raw_tip[0])), int(round(raw_tip[1]))),
            5,
            (255, 255, 255),
            -1
        )

    # Rekonstruierte Spitze einzeichnen
    if reconstructed_tip is not None:
        cv2.circle(
            result,
            (int(round(reconstructed_tip[0])), int(round(reconstructed_tip[1]))),
            9,
            (0, 255, 255),
            2
        )

    # Basis einzeichnen
    if base is not None:
        cv2.circle(
            result,
            (int(round(base[0])), int(round(base[1]))),
            6,
            (255, 255, 0),
            -1
        )

    # Achsen und Hilfspunkte einzeichnen
    if geometry is not None:
        cv2.line(
            result,
            tuple(geometry["left_end"].astype(int)),
            tuple(geometry["right_end"].astype(int)),
            (255, 255, 0),
            2
        )

        draw_points(result, geometry.get("middle_points"), (0, 255, 0))
        draw_points(result, geometry.get("edge_points"), (0, 255, 255))

        draw_text(
            result,
            f"Axis region: {geometry['axis_start']:.2f}-{geometry['axis_end']:.2f} | edge normal: {geometry['use_edge_normal']}",
            210
        )

    # Trajektorie einzeichnen
    for i in range(1, len(trajectory)):
        cv2.line(result, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

    # Statusinformationen anzeigen
    draw_text(result, f"Frame: {frame_idx}", 35)
    draw_text(result, f"Status: {status} | lost: {lost_counter}", 70)

    if geometry is not None:
        draw_text(
            result,
            f"Visible length: {geometry['visible_length']:.1f} | known: {known_length:.1f}",
            105
        )

    if taper_ratio is not None:
        draw_text(result, f"Taper ratio: {taper_ratio:.2f}", 140)

    draw_text(result, f"Occlusion mode: {occlusion_mode}", 175)

    if SAVE_VIDEO:
        out.write(result)

    if SHOW_VIDEO:
        cv2.imshow("Rightmost tip with shorter edge-normal prediction", result)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF

        if key == 27:
            break

    frame_idx += 1


# Ressourcen freigeben
cap.release()

if out is not None:
    out.release()

cv2.destroyAllWindows()

print(f"Gespeichert unter: {OUTPUT_VIDEO}")