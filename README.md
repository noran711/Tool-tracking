folder structure needed for the code to run:

-folder with your dataset
---images
---masks


  enter in line 9 your folder with your dataset (structure as discribed above)
  enter in line 10 the name for your folder with the augmented data 

  code takes all the images + the mask from the input folder, augments them and saves them in a new folder with subfolders images und masks

  enter in line 25-27 what transformations you want to have applied (p= probability of application)

  uncommetn line 63-73 if you want multiple images with transformations created from your input image

  line 75-78 creates only one copy of the input image with transformation 
