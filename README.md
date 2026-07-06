# Surgical Tool Tip Tracking

This project performs automatic **surgical instrument segmentation** and **tool tip detection** in endoscopic videos using a trained **nnU-Net v2** model.

The script:
- Extracts frames from an input video
- Runs inference with nnU-Net v2
- Detects the tool tip using skeletonization and endpoint analysis
- Generates:
  - a segmentation overlay video
  - a tool tip tracking video

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended)
- CUDA-compatible PyTorch installation

### Software
- Python 3.10+
- nnU-Net v2





### Input Video

```python
video_path = ".../test_videos/test_12.mp4"
```

### nnU-Net Configuration (enter the parameters of the wanted model)

```python
dataset_name = "Dataset101_Endo_own_finetuning"
config = "2d"
folds = ["0", "1", "2"]
trainer = "nnUNetTrainer"
plans = "nnUNetPlans"
device = "cuda"
```



The script generates two videos:

- **tool_tip_tracking_*.mp4** – original video with detected tool tip
- **prediction_overlay_*.mp4** – original video with segmentation overlay

Temporary frame and prediction folders are removed automatically after processing.
