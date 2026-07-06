This repository contains different approaches for **surgical instrument segmentation**, **tool tip detection**, and **tracking**. 


 **data_augmentation**
Contains Python scripts for image augmentation using **Albumentations**. These scripts can be used to generate augmented training data for segmentation models.


**ArUco**
Contains scripts for **tool tip tracking using ArUco markers** as well as example test videos. This approach serves as a marker-based reference for tool tracking.


**SAM2**
Contains a Python script for performing image segmentation using **Segment Anything Model 2 (SAM2)**


**tool_tracking_pen**
Contains experiments for tool tip tracking using a pen. This folder includes:
- nnU-Net training and inference files
- tracking scripts
- result files

 **most mature and stable version** of the tool tip tracking


**tracking_and_results**
Contains results and evaluation scripts from all three trained nnUNet models, including:
- nnU-Net-based segmentation
- Segmentation evaluation (Dice, IoU, Precision, Recall)
- Generated result videos and evaluation outputs
