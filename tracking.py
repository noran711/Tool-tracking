import os
import cv2
import glob
import shutil
import subprocess
import numpy as np

from skimage.morphology import skeletonize


BASE = os.path.expanduser("~/BHS_project")

os.environ["nnUNet_raw"] = os.path.join(BASE, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(BASE, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(BASE, "nnUNet_results")


# Input Video
video_path = os.path.join(BASE, "test_videos", "test_12.mp4")

# Output
results_dir = os.path.join(BASE, "predictions", "Dataset101", "Fold_0_1_2")

output_video = os.path.join(
    results_dir,
    "tool_tip_tracking_test_12.mp4"
)

output_overlay_video = os.path.join(
    results_dir,
    "prediction_overlay_test_12.mp4"
)

frames_dir = os.path.join(BASE, "temp_frames")
pred_dir = os.path.join(BASE, "temp_predictions")

dataset_name = "Dataset101_Endo_own_finetuning"
config = "2d"
folds = ["0", "1", "2"]
trainer = "nnUNetTrainer"
plans = "nnUNetPlans"
device = "cuda"

TARGET_CLASS = 1
OTHER_CLASS = 2

MASK_ALPHA = 0.5

os.makedirs(results_dir, exist_ok=True)

for d in [frames_dir, pred_dir]:

    if os.path.exists(d):
        shutil.rmtree(d)

    os.makedirs(d)

print("Lade Video...")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Video konnte nicht geladen werden: {video_path}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Extrahiere Frames...")

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_path = os.path.join(
        frames_dir,
        f"frame_{frame_idx:04d}_0000.png"
    )

    cv2.imwrite(frame_path, frame)

    frame_idx += 1

cap.release()

print(f"{frame_idx} Frames extrahiert.")

print("Starte nnUNet Prediction...")

command = [
    "nnUNetv2_predict",
    "-i", frames_dir,
    "-o", pred_dir,
    "-d", dataset_name,
    "-c", config,
    "-f", *folds,
    "-tr", trainer,
    "-p", plans,
    "-device", device,
    "-npp", "1",
    "-nps", "1"
]

subprocess.run(command, check=True)

print("Prediction abgeschlossen.")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Tooltip Video
out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# Overlay Video
overlay_out = cv2.VideoWriter(
    output_overlay_video,
    fourcc,
    fps,
    (width, height)
)

def is_endpoint(y, x, skel):

    neighborhood = skel[y-1:y+2, x-1:x+2]

    return np.sum(neighborhood) == 2


def touches_class2(y, x, class2_mask):

    return np.any(
        class2_mask[y-1:y+2, x-1:x+2]
    )

frame_files = sorted(
    glob.glob(
        os.path.join(frames_dir, "*.png")
    )
)


for idx, frame_path in enumerate(frame_files):

    frame = cv2.imread(frame_path)

    if frame is None:
        continue


    pred_path = os.path.join(
        pred_dir,
        f"frame_{idx:04d}.png"
    )

    pred = cv2.imread(
        pred_path,
        cv2.IMREAD_GRAYSCALE
    )


    if pred is None:

        print(f"Prediction fehlt: {pred_path}")

        out.write(frame)
        overlay_out.write(frame)

        continue

    
    overlay_frame = frame.copy()

    colored_mask = np.zeros_like(frame)

    # Klasse 1 -> Grün
    colored_mask[pred == TARGET_CLASS] = (0, 255, 0)

    # Klasse 2 -> Rot
    colored_mask[pred == OTHER_CLASS] = (0, 0, 255)

    overlay_frame = cv2.addWeighted(
        overlay_frame,
        1.0,
        colored_mask,
        MASK_ALPHA,
        0
    )

    # Konturen Klasse 1
    contours_target, _ = cv2.findContours(
        (pred == TARGET_CLASS).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay_frame,
        contours_target,
        -1,
        (0, 255, 0),
        2
    )

    # Konturen Klasse 2
    contours_other, _ = cv2.findContours(
        (pred == OTHER_CLASS).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay_frame,
        contours_other,
        -1,
        (0, 0, 255),
        2
    )

    overlay_out.write(overlay_frame)

    
    mask = (pred == TARGET_CLASS).astype(np.uint8)

    class2 = (pred == OTHER_CLASS).astype(np.uint8)

    skeleton = skeletonize(
        mask.astype(bool)
    ).astype(np.uint8)

    skel_coords = np.column_stack(
        np.where(skeleton == 1)
    )

    if skel_coords.size == 0:

        out.write(frame)

        continue

    h, w = skeleton.shape

    endpoints = []

    for y, x in skel_coords:

        if 1 <= y < h - 1 and 1 <= x < w - 1:

            if is_endpoint(y, x, skeleton):

                endpoints.append((y, x))

    if len(endpoints) == 0:

        out.write(frame)

        continue

    endpoints = np.array(endpoints)

    valid_endpoints = [

        (y, x)

        for (y, x) in endpoints

        if not touches_class2(
            y,
            x,
            class2
        )
    ]

    if len(valid_endpoints) == 0:

        out.write(frame)

        continue

    valid_endpoints = np.array(
        valid_endpoints
    )

    center = np.array([
        h / 2,
        w / 2
    ])

    distances = np.linalg.norm(
        valid_endpoints - center,
        axis=1
    )

    selected = valid_endpoints[
        np.argmin(distances)
    ]

    tooltip_frame = frame.copy()

    cv2.circle(
        tooltip_frame,
        (selected[1], selected[0]),
        10,
        (0, 0, 255),
        -1
    )

    out.write(tooltip_frame)

    print(f"Frame {idx}")


out.release()
overlay_out.release()


for d in [frames_dir, pred_dir]:

    if os.path.exists(d):
        shutil.rmtree(d)

print("Temp-Ordner gelöscht.")

print()
print("========================================")
print("FERTIG")
print(f"Tooltip Video: {output_video}")
print(f"Overlay Video: {output_overlay_video}")
print("========================================")