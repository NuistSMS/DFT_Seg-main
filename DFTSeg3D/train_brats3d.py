import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net.brats3d_model import BraTS3DSegModel
from prepare_brats3d import collect_text_rows, create_case_splits, inspect_h5_files, text_length_report
from utils.brats3d_dataset import BraTS3DDataset
from utils.brats3d_metrics import compute_case_metrics, summarize_case_metrics


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_prepared(cfg):
    output_dir = Path(cfg["EXPERIMENT"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    inspect_h5_files(cfg["DATA"]["h5_data_dir"], cfg["DATA"]["metadata_csv"], output_dir)
    text_df = collect_text_rows(cfg["DATA"]["text_dir"], cfg["TEXT"]["bert_type"])
    text_length_report(text_df, output_dir)
    return create_case_splits(cfg, text_df)


def write_environment(output_dir):
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "command": " ".join(sys.argv),
    }
    (output_dir / "environment.txt").write_text(json.dumps(info, indent=2), encoding="utf-8")


def move_batch(batch, device):
    return {
        "image": batch["image"].to(device),
        "image_low": batch["image_low"].to(device),
        "image_high": batch["image_high"].to(device),
        "label": batch["label"].to(device),
        "text": {k: v.to(device) for k, v in batch["text"].items()},
        "case_id": batch["case_id"],
        "volume": batch["volume"],
        "text_length": batch["text_length"],
    }


def dice_loss(logits, target, num_classes, include_background=False, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    start = 0 if include_background else 1
    probs = probs[:, start:]
    target_oh = target_oh[:, start:]
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * target_oh, dims)
    denom = torch.sum(probs + target_oh, dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits, target, cfg):
    ce = F.cross_entropy(logits, target)
    dice = dice_loss(logits, target, int(cfg["MODEL"]["num_classes"]))
    return float(cfg["TRAIN"]["ce_weight"]) * ce + float(cfg["TRAIN"]["dice_weight"]) * dice, ce.detach(), dice.detach()


@torch.no_grad()
def infer_logits(model, batch, cfg, device):
    image_low = batch["image_low"].to(device)
    image_high = batch["image_high"].to(device)
    text = {k: v.to(device) for k, v in batch["text"].items()}
    roi_size = tuple(int(x) for x in cfg["INFER"]["roi_size"])
    if image_low.shape[0] != 1:
        return model(image_low, image_high, text)

    image_pair = torch.cat([image_low, image_high], dim=1)

    def predictor(patch_pair):
        patch_low, patch_high = torch.chunk(patch_pair, chunks=2, dim=1)
        return model(patch_low, patch_high, text)

    return sliding_window_inference(
        image_pair,
        roi_size=roi_size,
        sw_batch_size=int(cfg["INFER"].get("sw_batch_size", 1)),
        predictor=predictor,
        overlap=float(cfg["INFER"].get("overlap", 0.25)),
        mode="gaussian",
    )


@torch.no_grad()
def evaluate(model, loader, cfg, device, split, output_dir=None, checkpoint_path="", include_surface=False):
    model.eval()
    total_loss = 0.0
    case_rows = []
    global_confusion = {}
    start_all = time.time()
    for batch in loader:
        batch = move_batch(batch, device)
        start = time.time()
        logits = infer_logits(model, batch, cfg, device)
        loss, _, _ = segmentation_loss(logits, batch["label"], cfg)
        total_loss += float(loss.detach().cpu())
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        target = batch["label"].detach().cpu().numpy()
        for i, case_id in enumerate(batch["case_id"]):
            row, confusion = compute_case_metrics(pred[i], target[i], case_id, include_surface=include_surface)
            row["checkpoint_path"] = checkpoint_path
            row["inference_seconds"] = time.time() - start
            row["prediction_shape"] = "x".join(map(str, pred[i].shape))
            case_rows.append(row)
            for name, vals in confusion.items():
                if name not in global_confusion:
                    global_confusion[name] = np.zeros(4, dtype=np.int64)
                global_confusion[name] += np.array(vals, dtype=np.int64)
    mean_loss = total_loss / max(1, len(loader))
    mean_dice_cols = [c for c in case_rows[0] if c.endswith("_Dice") and not c.startswith("background")] if case_rows else []
    finite_dice = [case[c] for case in case_rows for c in mean_dice_cols if np.isfinite(case[c])]
    mean_dice = float(np.mean(finite_dice)) if finite_dice else 0.0
    if output_dir:
        pd.DataFrame(case_rows).to_csv(output_dir / f"{split}_case_metrics.csv", index=False, encoding="utf-8-sig")
    return {"loss": mean_loss, "mean_dice": mean_dice, "case_rows": case_rows, "global_confusion": global_confusion, "seconds": time.time() - start_all}


def train(cfg, config_path):
    set_seed(int(cfg["EXPERIMENT"]["seed"]))
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(".hf_modules_cache").resolve()))
    output_dir = Path(cfg["EXPERIMENT"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    write_environment(output_dir)
    split_df = ensure_prepared(cfg)
    split_df.to_csv(output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")

    device_name = cfg["TRAIN"].get("device", "auto")
    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else ("cuda" if device_name == "cuda" else "cpu"))
    train_ds = BraTS3DDataset(Path(cfg["DATA"]["split_dir"]) / "train.csv", cfg, split="train", training=True)
    val_ds = BraTS3DDataset(Path(cfg["DATA"]["split_dir"]) / "val.csv", cfg, split="val", training=False)
    test_ds = BraTS3DDataset(Path(cfg["DATA"]["split_dir"]) / "test.csv", cfg, split="test", training=False)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["TRAIN"]["batch_size"]), shuffle=True, num_workers=int(cfg["TRAIN"]["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(cfg["TRAIN"]["num_workers"]))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=int(cfg["TRAIN"]["num_workers"]))

    model = BraTS3DSegModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(cfg), weight_decay=float(cfg["TRAIN"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["TRAIN"]["epochs"])))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["TRAIN"].get("amp", False)) and device.type == "cuda")
    writer = SummaryWriter(output_dir / "tensorboard")
    history = []
    best_metric = -1.0
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    wait = 0

    for epoch in range(int(cfg["TRAIN"]["epochs"])):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(batch["image_low"], batch["image_high"], batch["text"])
                loss, ce, dl = segmentation_loss(logits, batch["label"], cfg)
            scaler.scale(loss).backward()
            if float(cfg["TRAIN"].get("grad_clip", 0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["TRAIN"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            if int(cfg["TRAIN"].get("max_train_steps", 0)) and step + 1 >= int(cfg["TRAIN"]["max_train_steps"]):
                break

        scheduler.step()
        val_eval = evaluate(model, val_loader, cfg, device, "val", include_surface=False)
        train_loss = epoch_loss / max(1, min(len(train_loader), int(cfg["TRAIN"].get("max_train_steps", len(train_loader))) or len(train_loader)))
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_eval["loss"], "val_mean_dice": val_eval["mean_dice"], "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_eval["loss"], epoch)
        writer.add_scalar("dice/val_mean", val_eval["mean_dice"], epoch)
        print(json.dumps(row))

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "cfg": cfg,
        }
        torch.save(checkpoint, last_path)
        if val_eval["mean_dice"] > best_metric:
            best_metric = val_eval["mean_dice"]
            best_epoch = epoch
            checkpoint["best_metric"] = best_metric
            torch.save(checkpoint, best_path)
            wait = 0
        else:
            wait += 1
        if wait >= int(cfg["TRAIN"].get("patience", 999999)):
            break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    writer.close()

    test_summary_path = ""
    test_case_path = ""
    if bool(cfg["TEST"].get("run_test_after_train", True)):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_eval = evaluate(model, test_loader, cfg, device, "test", output_dir=output_dir, checkpoint_path=str(best_path), include_surface=True)
        metadata = {
            "dataset_name": "BraTS2020_TextBraTS_H5_slices",
            "experiment_name": cfg["EXPERIMENT"]["name"],
            "seed": int(cfg["EXPERIMENT"]["seed"]),
            "split_manifest": str(output_dir / "split_manifest.csv"),
            "train_cases": len(train_ds),
            "val_cases": len(val_ds),
            "test_cases": len(test_ds),
            "modalities": ",".join(cfg["DATA"]["channel_names"]),
            "input_channels": int(cfg["MODEL"]["in_channels"]),
            "wavelet_type": cfg["DATA"].get("wavelet_type", "haar"),
            "wavelet_branches": "low=aaa, high=sum(aad,ada,add,daa,dad,dda,ddd)",
            "patch_size": "x".join(map(str, cfg["DATA"]["patch_size"])),
            "model_name": "BraTS3DSegModel",
            "text_encoder": cfg["TEXT"]["bert_type"],
            "max_text_length": int(cfg["TEXT"]["max_text_length"]),
            "adapter_enabled": bool(cfg["TEXT"]["adapter_enabled"]),
            "adapter_bottleneck_dim": int(cfg["TEXT"]["adapter_bottleneck_dim"]),
            "adapter_dropout": float(cfg["TEXT"]["adapter_dropout"]),
            "best_epoch": int(best_epoch),
            "best_val_mean_dice": float(best_metric),
            "best_checkpoint_path": str(best_path),
            "config_path": str(output_dir / "config.yaml"),
            "log_path": str(output_dir),
            "test_status": "ok",
            "error_summary": "",
            "surface_metric_note": "HD95/ASD/ASSD are computed per case; non-finite one-empty cases are counted in *_nonfinite_excluded.",
        }
        summary = summarize_case_metrics(test_eval["case_rows"], test_eval["global_confusion"], metadata)
        summary_path = output_dir / "test_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
        test_case_path = str(output_dir / "test_case_metrics.csv")
        test_summary_path = str(summary_path)

    return {
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_mean_dice": best_metric,
        "training_history": str(output_dir / "training_history.csv"),
        "test_case_metrics": test_case_path,
        "test_summary": test_summary_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_brats3d_smoke.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = train(cfg, args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
