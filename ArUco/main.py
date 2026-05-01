import cv2
import cv2.aruco as aruco
import numpy as np
import time
from pathlib import Path

TARGET_ID = 0
MAX_LOST_FRAMES = 10
SPEED = 1.0

# Reale Markergröße: 15 mm
MARKER_LENGTH = 0.015  # Meter

# Abstand Markerzentrum -> Stiftspitze.
# Startwert aus altem Faktor: 15 mm * 5.5 = 82.5 mm
TIP_OFFSET = 0.14  # Meter

# Spitze relativ zum Marker-Koordinatensystem.
# Falls falsch: Vorzeichen ändern oder X/Y tauschen.
TIP_3D = np.array([[0, -TIP_OFFSET, 0]], dtype=np.float32)

# Alternativen testen, falls die Spitze auf der falschen Achse liegt:
# TIP_3D = np.array([[0, TIP_OFFSET, 0]], dtype=np.float32)
# TIP_3D = np.array([[TIP_OFFSET, 0, 0]], dtype=np.float32)
# TIP_3D = np.array([[-TIP_OFFSET, 0, 0]], dtype=np.float32)

# Stabilisierung: kleiner = schneller, größer = ruhiger
POSE_SMOOTHING = 0.35
MAX_ROT_JUMP_DEG = 60
MAX_TIP_JUMP_PX = 160

video_path = Path(__file__).parent / "test2.mp4"
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Video konnte nicht geöffnet werden: {video_path}")
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Näherungsweise Kameramatrix.
# Für präzises Tracking bitte durch echte Kamerakalibrierung ersetzen.
focal_length = frame_width
camera_matrix = np.array([
    [focal_length, 0, frame_width / 2],
    [0, focal_length, frame_height / 2],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

delay = int((1000 / video_fps) / SPEED)

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

half = MARKER_LENGTH / 2
object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float32)

# Kalman Filter für Stiftspitze: [x_tip, y_tip, vx, vy]
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

# Reaktionsfreudiger Kalman-Filter
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.5
kalman.errorCovPost = np.eye(4, dtype=np.float32)

initialized = False
trail = []
lost_frames = 0
prev_time = time.time()

pose_initialized = False
last_rvec = None
last_tvec = None
last_raw_tip = None


def rotation_jump_degrees(rvec_a, rvec_b):
    r_a, _ = cv2.Rodrigues(rvec_a)
    r_b, _ = cv2.Rodrigues(rvec_b)

    r_delta = r_a @ r_b.T
    cos_angle = (np.trace(r_delta) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def project_point(point_3d, rvec, tvec):
    projected, _ = cv2.projectPoints(
        point_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    return projected[0, 0]


def choose_stable_pose(c):
    global pose_initialized, last_rvec, last_tvec, last_raw_tip

    result = cv2.solvePnPGeneric(
        object_points,
        c,
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
            candidate_tip = project_point(TIP_3D, candidate_rvec, candidate_tvec)
            tip_jump = np.linalg.norm(candidate_tip - np.array(last_raw_tip))
            score += tip_jump * 0.8

        if best_score is None or score < best_score:
            best_score = score
            best_rvec = candidate_rvec
            best_tvec = candidate_tvec

    if best_rvec is None:
        return False, None, None

    return True, best_rvec, best_tvec


while True:
    ret, frame = cap.read()

    if not ret:
        print("Video fertig.")
        break

    now = time.time()
    tracker_dt = now - prev_time
    prev_time = now

    # Für Video-Dateien ist ein fixes dt stabiler als echte Rechenzeit.
    dt = 1.0 / video_fps
    tracker_fps = 1.0 / max(tracker_dt, 1e-6)

    kalman.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    prediction = kalman.predict()
    pred_x = int(prediction[0, 0])
    pred_y = int(prediction[1, 0])

    corners, ids, _ = detector.detectMarkers(frame)

    found = False
    angle = None

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            marker_id = int(ids[i][0])

            if marker_id != TARGET_ID:
                continue

            c = corners[i][0].astype(np.float32)

            success, rvec, tvec = choose_stable_pose(c)

            if not success:
                continue

            if pose_initialized:
                rot_jump = rotation_jump_degrees(rvec, last_rvec)

                if rot_jump <= MAX_ROT_JUMP_DEG:
                    rvec = POSE_SMOOTHING * last_rvec + (1.0 - POSE_SMOOTHING) * rvec
                    tvec = POSE_SMOOTHING * last_tvec + (1.0 - POSE_SMOOTHING) * tvec


            last_rvec = rvec.copy()
            last_tvec = tvec.copy()
            pose_initialized = True

            center_3d = np.array([[0, 0, 0]], dtype=np.float32)

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

            if last_raw_tip is not None:
                jump = np.linalg.norm(
                    np.array([tip_x, tip_y]) - np.array(last_raw_tip)
                )

                last_raw_tip = (tip_x, tip_y)

            else:
                last_raw_tip = (tip_x, tip_y)

            dx = tip_x - center_x
            dy = tip_y - center_y
            angle = np.degrees(np.arctan2(dy, dx))

            measurement = np.array([
                [np.float32(tip_x)],
                [np.float32(tip_y)]
            ])

            if not initialized:
                kalman.statePost = np.array([
                    [np.float32(tip_x)],
                    [np.float32(tip_y)],
                    [0.0],
                    [0.0]
                ], dtype=np.float32)
                initialized = True

            corrected = kalman.correct(measurement)

            smooth_tip_x = int(corrected[0, 0])
            smooth_tip_y = int(corrected[1, 0])

            found = True
            lost_frames = 0

            #cv2.drawFrameAxes(
            #    frame,
            #    camera_matrix,
            #    dist_coeffs,
            #    rvec,
            #    tvec,
            #    MARKER_LENGTH * 0.5
            #)

            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, "marker center",
                        (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1)

            cv2.circle(frame, (tip_x, tip_y), 6, (0, 165, 255), -1)
            cv2.putText(frame, "raw tip",
                        (tip_x + 10, tip_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 165, 255),
                        1)

            cv2.circle(frame, (smooth_tip_x, smooth_tip_y), 9, (0, 255, 0), -1)
            cv2.putText(frame,
                        f"tip | ID {marker_id}",
                        (smooth_tip_x + 10, smooth_tip_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2)

            #cv2.line(frame,
            #         (center_x, center_y),
            #         (smooth_tip_x, smooth_tip_y),
            #         (0, 255, 0),
            #         2)

            trail.append((smooth_tip_x, smooth_tip_y))
            break

    if not found and initialized:
        lost_frames += 1

        if lost_frames <= MAX_LOST_FRAMES:
            cv2.circle(frame, (pred_x, pred_y), 9, (255, 0, 0), -1)
            cv2.putText(frame,
                        "predicted tip",
                        (pred_x + 10, pred_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 0),
                        2)

    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], (255, 255, 0), 2)

    if len(trail) > 80:
        trail.pop(0)

    cv2.putText(frame,
                f"Playback FPS: {video_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Tracker FPS: {tracker_fps:.1f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    if angle is not None:
        cv2.putText(frame,
                    f"Angle: {angle:.1f} deg",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

    cv2.putText(frame,
                f"Marker: {MARKER_LENGTH * 1000:.1f} mm",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Tip offset: {TIP_OFFSET * 1000:.1f} mm",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                "ESC = quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2)

    cv2.imshow("Aruco Pen Tip Tracking", frame)

    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
