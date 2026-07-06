import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm



def load_mask(path):
    mask = np.array(Image.open(path))

    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask.astype(np.int32)



def compute_class_metrics(gt, pred, cls):
    gt_c = (gt == cls)
    pred_c = (pred == cls)

    tp = np.logical_and(gt_c, pred_c).sum()
    fp = np.logical_and(~gt_c, pred_c).sum()
    fn = np.logical_and(gt_c, ~pred_c).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return dice, iou, precision, recall



def evaluate_folders(gt_dir, pred_dir):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    gt_files = sorted(gt_dir.glob("*.png"))
    pred_files = sorted(pred_dir.glob("*.png"))

    gt_names = {f.name for f in gt_files}
    pred_names = {f.name for f in pred_files}

    common_files = sorted(gt_names & pred_names)

    results = []

    for name in tqdm(common_files, desc="Evaluating", unit="img"):
        gt = load_mask(gt_dir / name)
        pred = load_mask(pred_dir / name)

        classes = np.unique(np.concatenate([gt.flatten(), pred.flatten()]))

        for cls in classes:
            dice, iou, precision, recall = compute_class_metrics(gt, pred, cls)

            results.append({
                "filename": name,
                "class": int(cls),
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall
            })

    return results



def save_to_excel(results, output_path):
    df = pd.DataFrame(results)

    # --- Pivot Tables ---
    dice_df = df.pivot(index="filename", columns="class", values="dice")
    iou_df = df.pivot(index="filename", columns="class", values="iou")
    prec_df = df.pivot(index="filename", columns="class", values="precision")
    rec_df = df.pivot(index="filename", columns="class", values="recall")

    dice_df.columns = [f"dice_class{c}" for c in dice_df.columns]
    iou_df.columns = [f"iou_class{c}" for c in iou_df.columns]
    prec_df.columns = [f"precision_class{c}" for c in prec_df.columns]
    rec_df.columns = [f"recall_class{c}" for c in rec_df.columns]

    
    mean_dice_all = dice_df.mean(axis=1)
    mean_iou_all = iou_df.mean(axis=1)
    mean_prec_all = prec_df.mean(axis=1)
    mean_rec_all = rec_df.mean(axis=1)

    
    def drop_bg(df):
        return df.loc[:, ~df.columns.str.contains("class0")]

    mean_dice_no_bg = drop_bg(dice_df).mean(axis=1)
    mean_iou_no_bg = drop_bg(iou_df).mean(axis=1)
    mean_prec_no_bg = drop_bg(prec_df).mean(axis=1)
    mean_rec_no_bg = drop_bg(rec_df).mean(axis=1)

    
    final_df = pd.concat([
        dice_df,
        iou_df,
        prec_df,
        rec_df,

        mean_dice_all.rename("mean_dice_all"),
        mean_iou_all.rename("mean_iou_all"),
        mean_prec_all.rename("mean_precision_all"),
        mean_rec_all.rename("mean_recall_all"),

        mean_dice_no_bg.rename("mean_dice_no_bg"),
        mean_iou_no_bg.rename("mean_iou_no_bg"),
        mean_prec_no_bg.rename("mean_precision_no_bg"),
        mean_rec_no_bg.rename("mean_recall_no_bg"),
    ], axis=1)

    mean_row = final_df.mean(numeric_only=True)
    mean_row.name = "GLOBAL_MEAN"

    final_df = pd.concat([final_df, mean_row.to_frame().T])

    final_df = final_df.reset_index().rename(columns={"index": "filename"})

    final_df.to_excel(output_path, index=False)

    print(f"\nExcel gespeichert unter: {output_path}")



if __name__ == "__main__":
    gt_dir = r"test_12/test_12_gt_masks"
    pred_dir = r"Dataset102_Endo_own_SAM/masks"

    output_excel = r"Dataset102_Endo_own_SAM/results.xlsx"

    results = evaluate_folders(gt_dir, pred_dir)
    save_to_excel(results, output_excel)