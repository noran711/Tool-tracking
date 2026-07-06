Due to the file size, the nnUNet_results folder could not be uploaded here to GitHub, so it can be found in the Google Drive folder under Results/Resutls_Nora


# Instruction

1. Check and adjust the configuration
   - Edit [Tool_tracking_pen/scripts/config.py](Tool_tracking_pen/scripts/config.py) so that the paths to the data, models, and output folders match your setup.

2. Convert raw data to the nnU-Net format
   - Run [Tool_tracking_pen/scripts/dataset/convert_pen_to_nnunet.py](Tool_tracking_pen/scripts/dataset/convert_pen_to_nnunet.py).
   - This script reads the images and the corresponding masks from the dataset folders and creates the structure required for nnU-Net.

3. Prepare images
   - Use [Tool_tracking_pen/resize_images.py](Tool_tracking_pen/resize_images.py) if images in the folder predictions/input need to be prepared.
   - It resizes the images to the target size 432x432 and saves them as PNG files.

4. Train the model
   - Set the nnU-Net environment variables.
   - Before training, the nnU-Net paths have to be set:

   set nnUNet_raw=Path\to\Tool_tracking_pen\nnUNet_raw
   set nnUNet_preprocessed=Path\to\Tool_tracking_pen\nnUNet_preprocessed
   set nnUNet_results=Path\to\Tool_tracking_pen\nnUNet_results

   Then the dataset is checked and prepared by nnU-Net:
   - nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

   Train the model:
   - nnUNetv2_train 100 2d 0 -tr nnUNetTrainer_100epochs

5. Run predictions with the segmentation model
   - For single images: [Tool_tracking_pen/scripts/inference/predict.py](Tool_tracking_pen/scripts/inference/predict.py)
   - For video frames: [Tool_tracking_pen/scripts/tracking/predict_.video_frames.py](Tool_tracking_pen/scripts/tracking/predict_.video_frames.py)
   - Both scripts generate masks from the input images and save them in the prediction folders.

6. Visualize or track the results
   - [Tool_tracking_pen/scripts/tracking/visualize_frames.py](Tool_tracking_pen/scripts/tracking/visualize_frames.py) shows the tracking results or the evaluation of the frames.
   - [Tool_tracking_pen/scripts/tracking/visualize_verdeckte_frames.py](Tool_tracking_pen/scripts/tracking/visualize_verdeckte_frames.py) is intended for the special case of occluded or partially visible tips.
