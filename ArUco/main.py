import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path


# Zielmarker und erlaubte verlorene Frames
TARGET_ID = 0
MAX_LOST_FRAMES = 10

# Markergröße und Abstand der Spitze vom Marker
MARKER_LENGTH = 0.015
TIP_OFFSET = 0.14
TIP_3D = np.array([[0, -TIP_OFFSET, 0]], dtype=np.float32)

# Glättung der Pose
POSE_SMOOTHING = 0.35
MAX_ROT_JUMP_DEG = 60

# Anzeigeeinstellungen
SHOW_TRAIL = True
TRAIL_LENGTH = 50
TIP_RADIUS = 6
TRAIL_THICKNESS = 2

SHOW_MARKER = True
SHOW_CENTER = True
SHOW_DIRECTION = True

# Ein- und Ausgabevideo
BASE_DIR = Path(__file__).parent
VIDEO_PATH = BASE_DIR / "test2.mp4"
OUTPUT_PATH = BASE_DIR / "test2_tracked_tip_visualized.mp4"
MAX_DURATION_SECONDS = 10


# Video öffnen
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Video konnte nicht geoeffnet werden: {VIDEO_PATH}")

# Videoinformationen auslesen
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30.0

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ausgabevideo vorbereiten
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    video_fps,
    (frame_width, frame_height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Ausgabevideo konnte nicht erstellt werden: {OUTPUT_PATH}")


# Vereinfachte Kameramatrix
focal_length = frame_width
camera_matrix = np.array([
    [focal_length, 0, frame_width / 2],
    [0, focal_length, frame_height / 2],
    [0, 0, 1]
], dtype=np.float32)

# Keine Linsenverzerrung angenommen
dist_coeffs = np.zeros((5, 1), dtype=np.float32)


# ArUco-Detektor einstellen
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.01
params.adaptiveThreshWinSizeMin = 5
params.adaptiveThreshWinSizeMax = 21
params.adaptiveThreshWinSizeStep = 8
params.minMarkerPerimeterRate = 0.03
params.maxMarkerPerimeterRate = 4.0
params.polygonalApproxAccuracyRate = 0.03
params.minCornerDistanceRate = 0.03
params.minDistanceToBorder = 3

detector = aruco.ArucoDetector(dictionary, params)

# 3D-Eckpunkte des Markers
half = MARKER_LENGTH / 2
object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float32)


# Kalman-Filter für die Spitze
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.5
kalman.errorCovPost = np.eye(4, dtype=np.float32)

dt = 1.0 / video_fps
kalman.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


# Zustandsvariablen
initialized = False
pose_initialized = False

last_rvec = None
last_tvec = None
last_raw_tip = None

lost_frames = 0
trail = []


def rotation_jump_degrees(rvec_a, rvec_b):
    # Rotationssprung zwischen zwei Posen berechnen
    r_a, _ = cv2.Rodrigues(rvec_a)
    r_b, _ = cv2.Rodrigues(rvec_b)

    r_delta = r_a @ r_b.T
    cos_angle = (np.trace(r_delta) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def project_point(point_3d, rvec, tvec):
    # 3D-Punkt ins Bild projizieren
    projected, _ = cv2.projectPoints(
        point_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    return projected[0, 0]


def choose_stable_pose(corners_2d):
    # Stabilste Pose aus den möglichen Lösungen wählen
    global pose_initialized, last_rvec, last_raw_tip

    result = cv2.solvePnPGeneric(
        object_points,
        corners_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    success = result[0]
    rvecs = result[1]
    tvecs = result[2]
    reprojection_errors = result[3] if len(result) > 3 else None

    if not success or len(rvecs) == 0:
        return False, None, None

    best_score = None
    best_rvec = None
    best_tvec = None

    for idx, (candidate_rvec, candidate_tvec) in enumerate(zip(rvecs, tvecs)):
        if candidate_tvec[2, 0] <= 0:
            continue

        score = 0.0

        if reprojection_errors is not None:
            score += float(np.ravel(reprojection_errors)[idx]) * 50.0

        if pose_initialized:
            score += rotation_jump_degrees(candidate_rvec, last_rvec) * 1.5

        if last_raw_tip is not None:
            candidate_tip = project_point(
                TIP_3D,
                candidate_rvec,
                candidate_tvec
            )
            score += np.linalg.norm(
                candidate_tip - np.array(last_raw_tip)
            ) * 0.8

        if best_score is None or score < best_score:
            best_score = score
            best_rvec = candidate_rvec
            best_tvec = candidate_tvec

    if best_rvec is None:
        return False, None, None

    return True, best_rvec, best_tvec


# Maximale Frameanzahl festlegen
max_frames = int(video_fps * MAX_DURATION_SECONDS)
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret or frame_count >= max_frames:
        break

    frame_count += 1

    # Nächste Position vorhersagen
    kalman.predict()

    # Marker im aktuellen Frame suchen
    corners, ids, _ = detector.detectMarkers(frame)

    found = False
    display_tip = None
    display_center = None
    display_marker_corners = None
    display_marker_id = None

    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])

            if marker_id != TARGET_ID:
                continue

            c = corners[i][0].astype(np.float32)

            success, rvec, tvec = choose_stable_pose(c)

            if not success:
                continue

            # Pose glätten
            if pose_initialized:
                rot_jump = rotation_jump_degrees(rvec, last_rvec)

                if rot_jump <= MAX_ROT_JUMP_DEG:
                    rvec = (
                        POSE_SMOOTHING * last_rvec
                        + (1.0 - POSE_SMOOTHING) * rvec
                    )
                    tvec = (
                        POSE_SMOOTHING * last_tvec
                        + (1.0 - POSE_SMOOTHING) * tvec
                    )

            last_rvec = rvec.copy()
            last_tvec = tvec.copy()
            pose_initialized = True

            center_3d = np.array([[0, 0, 0]], dtype=np.float32)

            # Markerzentrum und Spitze projizieren
            projected_center, _ = cv2.projectPoints(
                center_3d,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )

            projected_tip, _ = cv2.projectPoints(
                TIP_3D,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )

            center_x, center_y = projected_center[0, 0].astype(int)
            tip_x, tip_y = projected_tip[0, 0].astype(int)

            last_raw_tip = (tip_x, tip_y)

            measurement = np.array([
                [np.float32(tip_x)],
                [np.float32(tip_y)]
            ])

            # Kalman-Filter initialisieren
            if not initialized:
                kalman.statePost = np.array([
                    [np.float32(tip_x)],
                    [np.float32(tip_y)],
                    [0.0],
                    [0.0]
                ], dtype=np.float32)
                initialized = True

            # Messung korrigieren
            corrected = kalman.correct(measurement)

            smooth_tip_x = int(corrected[0, 0])
            smooth_tip_y = int(corrected[1, 0])

            display_tip = (smooth_tip_x, smooth_tip_y)
            display_center = (center_x, center_y)
            display_marker_corners = [corners[i]]
            display_marker_id = ids[i:i + 1]

            found = True
            lost_frames = 0
            break

    # Keine gültige Markererkennung
    if not found and initialized:
        lost_frames += 1
        display_tip = None

    if display_tip is not None:
        tip_x, tip_y = display_tip

        if 0 <= tip_x < frame_width and 0 <= tip_y < frame_height:
            # Marker einzeichnen
            if display_marker_corners is not None and SHOW_MARKER:
                aruco.drawDetectedMarkers(
                    frame,
                    display_marker_corners,
                    display_marker_id
                )

            # Mittelpunkt einzeichnen
            if display_center is not None and SHOW_CENTER:
                cv2.circle(
                    frame,
                    display_center,
                    5,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA
                )

            # Richtung zur Spitze einzeichnen
            if display_center is not None and SHOW_DIRECTION:
                cv2.arrowedLine(
                    frame,
                    display_center,
                    display_tip,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.08
                )

            # Bewegungsspur speichern
            trail.append(display_tip)

            if len(trail) > TRAIL_LENGTH:
                trail.pop(0)

            # Bewegungsspur einzeichnen
            if SHOW_TRAIL and len(trail) > 1:
                for j in range(1, len(trail)):
                    alpha = j / len(trail)
                    thickness = max(1, int(TRAIL_THICKNESS * alpha))
                    cv2.line(
                        frame,
                        trail[j - 1],
                        trail[j],
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )

            # Spitze einzeichnen
            cv2.circle(
                frame,
                display_tip,
                TIP_RADIUS + 3,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )
            cv2.circle(
                frame,
                display_tip,
                TIP_RADIUS,
                (0, 165, 255),
                -1,
                cv2.LINE_AA
            )

    writer.write(frame)


# Ressourcen freigeben
cap.release()
writer.release()

print(f"Fertig. Das getrackte Video wurde gespeichert unter:\n{OUTPUT_PATH}")
