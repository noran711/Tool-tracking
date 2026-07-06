# Segmentation Evaluation

This script evaluates semantic segmentation results by comparing predicted masks with ground truth masks.

For every image and every detected class, the following metrics are computed:

- Dice Coefficient
- Intersection over Union (IoU)
- Precision
- Recall

Additionally, the script calculates:

- Mean metrics across all classes
- Mean metrics excluding the background class (Class 0)
- Global mean values over the entire dataset

All results are exported to a single Excel file.


## User Configuration

Adjust the following paths before running the script.

### Ground Truth Masks

```python
gt_dir = "test_12/test_12_gt_masks"
```

Directory containing the ground truth segmentation masks.

### Predicted Masks

```python
pred_dir = "Dataset102_Endo_own_SAM/masks"
```

Directory containing the predicted segmentation masks.

### Output File

```python
output_excel = "Dataset102_Endo_own_SAM/results.xlsx"
```

Path where the evaluation results will be saved.
