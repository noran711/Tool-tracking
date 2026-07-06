"""
Randomly sample frame/mask pairs from the SAM2 segmentation output and copy
them into imagesTs/labelsTs using nnU-Net naming conventions.

Source folders (left untouched):
  output/frames/frame_XXXXX.jpg
  output/masks/frame_XXXXX.png

Created folders:
  imagesTs/img_NNN_0000.png
  labelsTs/img_NNN.png
"""

import os
import random
import shutil

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "output", "frames")
MASKS_DIR = os.path.join(BASE_DIR, "output", "masks")

IMAGES_TS_DIR = os.path.join(BASE_DIR, "imagesTs")
LABELS_TS_DIR = os.path.join(BASE_DIR, "labelsTs")

NUM_SAMPLES = 200  # how many frame/mask pairs to randomly select
SEED = 42

os.makedirs(IMAGES_TS_DIR, exist_ok=True)
os.makedirs(LABELS_TS_DIR, exist_ok=True)

# Match frames and masks by their shared "frame_XXXXX" index
frame_indices = {
    f.split("_")[1].split(".")[0]
    for f in os.listdir(FRAMES_DIR)
    if f.startswith("frame_") and f.endswith(".jpg")
}
mask_indices = {
    f.split("_")[1].split(".")[0]
    for f in os.listdir(MASKS_DIR)
    if f.startswith("frame_") and f.endswith(".png")
}

common_indices = sorted(frame_indices & mask_indices)
print(f"Found {len(common_indices)} matching frame/mask pairs")

random.seed(SEED)
num_to_select = min(NUM_SAMPLES, len(common_indices))
selected = sorted(random.sample(common_indices, num_to_select))

for new_idx, frame_idx in enumerate(selected, start=1):
    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx}.jpg")
    mask_path = os.path.join(MASKS_DIR, f"frame_{frame_idx}.png")

    img = Image.open(frame_path).convert("RGB")
    img.save(os.path.join(IMAGES_TS_DIR, f"img_{new_idx:03d}_0000.png"))

    shutil.copy(mask_path, os.path.join(LABELS_TS_DIR, f"img_{new_idx:03d}.png"))

print(f"Selected {len(selected)} pairs")
print(f"Images saved to: {IMAGES_TS_DIR}")
print(f"Labels saved to: {LABELS_TS_DIR}")
