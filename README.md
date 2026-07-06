# Surgical Instrument Segmentation and Tool Tip Tracking

This repository contains different approaches for **surgical instrument segmentation**, **tool tip detection**, and **tracking**.

---

## 📁 data_augmentation

Contains Python scripts for image augmentation using **Albumentations**. These scripts can be used to generate augmented training data for segmentation models.

---

## 📁 ArUco

Contains scripts for **tool tip tracking using ArUco markers** together with example test videos. This approach serves as a marker-based reference for tool tracking.

---

## 📁 SAM2

Contains a Python script for performing image segmentation using the **Segment Anything Model 2 (SAM2)**.

---

## 📁 tool_tracking_pen

Contains experiments for **tool tip tracking using a pen**.

This folder includes:
- nnU-Net training and inference files
- Tool tip tracking scripts
- Result files

This is the **most mature and stable implementation** of the tool tip tracking pipeline.

---

## 📁 tracking_and_results

Contains the final tracking results and evaluation scripts for all three trained **nnU-Net** models, including:

- nnU-Net-based segmentation
- Segmentation evaluation (**Dice, IoU, Precision, Recall**)
- Evaluation outputs
