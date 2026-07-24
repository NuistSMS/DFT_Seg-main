import argparse
import os
import warnings
import logging

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_MODULES_CACHE', os.path.abspath('.hf_modules_cache'))
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import SegData
import utils.config as config
from net.creratemodel import CreateModel
from medpy.metric.binary import hd95, assd
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description='Test Breast Segmentation')
    parser.add_argument('--config', default='./config/train_breast.yaml', type=str)
    parser.add_argument('--checkpoint', default='./save_model/breast.ckpt', type=str)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg, args.checkpoint


def compute_metrics(preds_list, gts_list):
    """
    preds_list: list of numpy arrays, each shape (H, W), binary 0/1
    gts_list:   list of numpy arrays, each shape (H, W), binary 0/1

    Returns dict with:
      - global_dice:   sum(intersection)*2 / sum(pred+gt)  across all pixels
      - global_iou:    sum(intersection) / sum(union)       across all pixels
      - per_img_dice:  mean of per-image dice
      - per_img_iou:   mean of per-image IoU
      - hd95:          mean of per-image HD95
      - assd:          mean of per-image ASSD
    """
    n = len(preds_list)

    # --- Global accumulators ---
    total_intersection = 0.0
    total_union = 0.0
    total_pred_sum = 0.0
    total_gt_sum = 0.0

    # --- Per-image accumulators ---
    per_dice = []
    per_iou = []
    per_hd95 = []
    per_assd = []

    for pred, gt in zip(preds_list, gts_list):
        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)

        intersection = np.count_nonzero(pred_bool & gt_bool)
        union = np.count_nonzero(pred_bool | gt_bool)
        pred_sum = np.count_nonzero(pred_bool)
        gt_sum = np.count_nonzero(gt_bool)

        # Global accumulators
        total_intersection += intersection
        total_union += union
        total_pred_sum += pred_sum
        total_gt_sum += gt_sum

        # Per-image dice
        denom = pred_sum + gt_sum
        if denom > 0:
            per_dice.append(2.0 * intersection / denom)
        else:
            per_dice.append(1.0)  # both empty → perfect match

        # Per-image IoU
        if union > 0:
            per_iou.append(intersection / union)
        else:
            per_iou.append(1.0)

        # HD95 & ASSD — needs at least one foreground pixel on both sides
        if pred_sum > 0 and gt_sum > 0:
            per_hd95.append(hd95(pred_bool, gt_bool))
            per_assd.append(assd(pred_bool, gt_bool))
        elif pred_sum == 0 and gt_sum == 0:
            per_hd95.append(0.0)
            per_assd.append(0.0)
        # else one side empty → skip distance metrics (no valid surface)

    # --- Global metrics ---
    if total_pred_sum + total_gt_sum > 0:
        global_dice = 2.0 * total_intersection / (total_pred_sum + total_gt_sum)
    else:
        global_dice = 1.0

    if total_union > 0:
        global_iou = total_intersection / total_union
    else:
        global_iou = 1.0

    # --- Per-image averages ---
    results = {
        'global_dice': global_dice,
        'global_iou': global_iou,
        'per_img_dice_mean': np.mean(per_dice) if per_dice else 0.0,
        'per_img_dice_std': np.std(per_dice) if per_dice else 0.0,
        'per_img_iou_mean': np.mean(per_iou) if per_iou else 0.0,
        'per_img_iou_std': np.std(per_iou) if per_iou else 0.0,
        'hd95_mean': np.mean(per_hd95) if per_hd95 else 0.0,
        'hd95_std': np.std(per_hd95) if per_hd95 else 0.0,
        'assd_mean': np.mean(per_assd) if per_assd else 0.0,
        'assd_std': np.std(per_assd) if per_assd else 0.0,
        'n_hd95_valid': len(per_hd95),
        'n_total': n,
    }
    return results


if __name__ == '__main__':
    cfg, ckpt_path = get_parser()

    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = CreateModel(cfg)
    checkpoint = torch.load(ckpt_path, map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.\n")

    # 测试集 dataloader
    ds_test = SegData(dataname='breast',
                      csv_path=cfg.test_csv_path,
                      root_path=cfg.test_root_path,
                      tokenizer=cfg.bert_type,
                      image_size=cfg.image_size,
                      mode='test',
                      max_text_len=getattr(cfg, 'max_text_len', 24))

    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4)

    # 收集所有预测和目标
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Testing"):
            x, y = batch
            # x = [image, image2, text_dict]
            if isinstance(x, list):
                x[0] = x[0].to(device)
                x[1] = x[1].to(device)
                x[2]['input_ids'] = x[2]['input_ids'].to(device)
                x[2]['attention_mask'] = x[2]['attention_mask'].to(device)

            # model.forward(x) → (preds, preds2, _) where preds == preds2
            preds, _, _ = model(x)
            # preds: (1, 1, H, W) sigmoid output
            pred_binary = (preds > 0.5).int().cpu().numpy().squeeze()  # (H, W)

            if isinstance(y, torch.Tensor):
                gt_np = y.int().cpu().numpy().squeeze()
            else:
                gt_np = y

            all_preds.append(pred_binary)
            all_gts.append(gt_np)

    # 计算指标
    results = compute_metrics(all_preds, all_gts)

    print("\n" + "=" * 60)
    print("              FINAL TEST RESULTS                  ")
    print("=" * 60)
    print(f" Global Dice     : {results['global_dice']:.4f}")
    print(f" Global IoU      : {results['global_iou']:.4f}")
    print(f" Per-Image Dice  : {results['per_img_dice_mean']:.4f} ± {results['per_img_dice_std']:.4f}")
    print(f" Per-Image IoU   : {results['per_img_iou_mean']:.4f} ± {results['per_img_iou_std']:.4f}")
    print(f" HD95            : {results['hd95_mean']:.4f} ± {results['hd95_std']:.4f}  "
          f"(valid: {results['n_hd95_valid']}/{results['n_total']})")
    print(f" ASSD            : {results['assd_mean']:.4f} ± {results['assd_std']:.4f}  "
          f"(valid: {results['n_hd95_valid']}/{results['n_total']})")
    print("=" * 60 + "\n")
