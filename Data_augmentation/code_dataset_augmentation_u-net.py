import os
import random
import shutil
import cv2
import albumentations as A
from pathlib import Path

#fill your input source and output folder name here
input_folder = "input/dataset_raw"
output_folder_name = "dataset_augmented"

#create output folder structure for augmented data
def create_output_folder_structure(output_folder):

    output = Path(output_folder)

    output.mkdir(parents=True, exist_ok=True)

    subfolders = ["images", "masks"]

    for subfolder in subfolders:
        (output / subfolder).mkdir(parents=True, exist_ok=True)

#define augmentation of data, p defines probability of application of transformation
transform = A.Compose([
    A.ElasticTransform(alpha=150, sigma=150, p=1.0), 
    A.Rotate(limit=56, p=1.0)
])

#apply augmentation on dataset
def data_augmentation(input_folder, output_folder, transform):

    #read in input folder path and subfolders that contain data
    input_path = Path(input_folder)
    images_path = input_path / "images"
    masks_path = input_path / "masks"

    output = Path(output_folder)
    output_images = output / "images"
    output_masks = output / "masks"

    #iterate over all images in dataset input folder
    for image_file in images_path.iterdir():
        if not image_file.is_file():
            continue 

        #check if corresponding mask exists to image
        mask_file = masks_path / f"{image_file.stem}_mask.png"

        if not mask_file.exists():
            #print(f"Mask fehlt für: {image_file.name}")
            continue

        #read in image and mask
        image = cv2.imread(str(image_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        # save originals in augmented dataset folders 
        orig_name = image_file.name
        cv2.imwrite(str(output_images / orig_name), image)
        cv2.imwrite(str(output_masks / orig_name), mask)
    
        # #generates 3 images with the transformation applied
        # for i in range(3):
        #     augmented = transform(image=image, mask=mask)
        #     augmented_image = augmented["image"]
        #     augmented_mask = augmented["mask"]

        #     new_name = f"{image_file.stem}_aug_{i}{image_file.suffix}"

        #     #save augmented image and mask
        #     cv2.imwrite(str(output_images / new_name), augmented_image)
        #     cv2.imwrite(str(output_masks/ new_name), augmented_mask)
        
        #generates 1 image with transformation applied 
        augmented = transform(image=image, mask=mask)
        augmented_image = augmented["image"]
        augmented_mask = augmented["mask"]

        #give augmented image or mask new name/ add on 
        new_name = f"{image_file.stem}_aug{image_file.suffix}"

        #save augmented image and mask
        cv2.imwrite(str(output_images / new_name), augmented_image)
        cv2.imwrite(str(output_masks / new_name), augmented_mask)
        

current_dir = Path(__file__).parent
input_path = current_dir / input_folder
output_path = current_dir / "input" / output_folder_name

create_output_folder_structure(output_path )
data_augmentation(input_path, output_path, transform)