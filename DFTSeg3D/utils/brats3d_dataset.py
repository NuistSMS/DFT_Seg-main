from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _case_volume(case_id):
    return int(str(case_id).split("_")[-1])


def one_hot_mask_to_label(mask):
    """Convert H5 mask [D,H,W,3] to class labels [D,H,W].

    The inspected H5 files store three disjoint binary channels for non-background
    BraTS labels. We map them to contiguous labels for CrossEntropy:
    0 background, 1 channel_0/NCR-NET, 2 channel_1/edema, 3 channel_2/enhancing.
    """
    label = np.zeros(mask.shape[:3], dtype=np.int64)
    for channel in range(mask.shape[-1]):
        label[mask[..., channel] > 0] = channel + 1
    return label


def load_case_volume(h5_dir, volume, image_key="image", mask_key="mask"):
    h5_dir = Path(h5_dir)
    images = []
    masks = []
    for slice_idx in range(155):
        path = h5_dir / f"volume_{int(volume)}_slice_{slice_idx}.h5"
        if not path.exists():
            raise FileNotFoundError(f"Missing H5 slice: {path}")
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            if image_key not in keys or mask_key not in keys:
                raise KeyError(f"{path} keys={sorted(keys)}, expected image_key={image_key}, mask_key={mask_key}")
            image = f[image_key][()]
            mask = f[mask_key][()]
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask spatial mismatch in {path}: {image.shape} vs {mask.shape}")
        images.append(image.astype(np.float32))
        masks.append(mask.astype(np.uint8))

    image = np.stack(images, axis=0)  # [D,H,W,C]
    mask = np.stack(masks, axis=0)    # [D,H,W,3]
    label = one_hot_mask_to_label(mask)
    image = np.moveaxis(image, -1, 0)  # [C,D,H,W]
    return image, label


def normalize_modalities(image, nonzero=True, eps=1e-6):
    out = image.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        arr = out[c]
        region = arr != 0 if nonzero else np.ones_like(arr, dtype=bool)
        if not np.any(region):
            continue
        mean = float(arr[region].mean())
        std = float(arr[region].std())
        out[c] = (arr - mean) / max(std, eps)
        if nonzero:
            out[c][~region] = 0
    return out


def resize_volume_like(image, spatial_size):
    tensor = torch.from_numpy(image[None]).float()
    tensor = F.interpolate(tensor, size=tuple(int(x) for x in spatial_size), mode="trilinear", align_corners=False)
    return tensor.numpy()[0].astype(np.float32)


def wavelet_decompose_3d(image, wavelet_type="haar"):
    """3D version of the original low/high wavelet split.

    The 2D code uses LL as low frequency and LH+HL+HH as high frequency.
    Here `aaa` is the 3D low-frequency volume and all other seven 3D
    detail bands are summed as the high-frequency volume, then resized
    back to the same D/H/W as the label.
    """
    lows = []
    highs = []
    spatial_size = image.shape[1:]
    for channel in range(image.shape[0]):
        coeffs = pywt.dwtn(image[channel], wavelet=wavelet_type, axes=(0, 1, 2))
        low = coeffs["aaa"].astype(np.float32)
        high = np.zeros_like(low, dtype=np.float32)
        for key, value in coeffs.items():
            if key != "aaa":
                high += value.astype(np.float32)
        lows.append(low)
        highs.append(high)

    low = resize_volume_like(np.stack(lows, axis=0), spatial_size)
    high = resize_volume_like(np.stack(highs, axis=0), spatial_size)
    return normalize_modalities(low, nonzero=False), normalize_modalities(high, nonzero=False)


def random_intensity_aug(image, rng):
    if rng.random() < 0.25:
        image = image + rng.normal(0, 0.05, size=image.shape).astype(np.float32)
    if rng.random() < 0.25:
        gamma = rng.uniform(0.8, 1.2)
        sign = np.sign(image)
        image = sign * (np.abs(image) ** gamma)
    if rng.random() < 0.25:
        image = image * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    return image


def random_flip_3d(image_low, image_high, label, rng):
    for axis in [1, 2, 3]:
        if rng.random() < 0.5:
            image_low = np.flip(image_low, axis=axis).copy()
            image_high = np.flip(image_high, axis=axis).copy()
            label = np.flip(label, axis=axis - 1).copy()
    return image_low, image_high, label


def crop_patch_pair(image_low, image_high, label, patch_size, rng, foreground_prob=0.8):
    c, d, h, w = image_low.shape
    pd, ph, pw = patch_size
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d or pad_h or pad_w:
        image_low_t = torch.from_numpy(image_low[None])
        image_high_t = torch.from_numpy(image_high[None])
        label_t = torch.from_numpy(label[None, None].astype(np.float32))
        image_low = F.pad(image_low_t, (0, pad_w, 0, pad_h, 0, pad_d)).numpy()[0]
        image_high = F.pad(image_high_t, (0, pad_w, 0, pad_h, 0, pad_d)).numpy()[0]
        label = F.pad(label_t, (0, pad_w, 0, pad_h, 0, pad_d)).numpy()[0, 0].astype(np.int64)
        c, d, h, w = image_low.shape

    use_fg = rng.random() < foreground_prob and np.any(label > 0)
    if use_fg:
        coords = np.argwhere(label > 0)
        center = coords[rng.integers(0, len(coords))]
        start_d = int(np.clip(center[0] - rng.integers(0, pd), 0, d - pd))
        start_h = int(np.clip(center[1] - rng.integers(0, ph), 0, h - ph))
        start_w = int(np.clip(center[2] - rng.integers(0, pw), 0, w - pw))
    else:
        start_d = int(rng.integers(0, max(1, d - pd + 1)))
        start_h = int(rng.integers(0, max(1, h - ph + 1)))
        start_w = int(rng.integers(0, max(1, w - pw + 1)))

    return (
        image_low[:, start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw],
        image_high[:, start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw],
        label[start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw],
    )


class BraTS3DDataset(Dataset):
    def __init__(self, split_csv, cfg, split, training=False):
        self.rows = pd.read_csv(split_csv)
        self.split = split
        self.training = training
        self.data_cfg = cfg["DATA"]
        self.text_cfg = cfg["TEXT"]
        limit = self.data_cfg.get(f"{split}_cases_limit")
        if limit:
            self.rows = self.rows.iloc[: int(limit)].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_cfg["bert_type"], trust_remote_code=True)
        self.max_text_length = int(self.text_cfg["max_text_length"])
        self.patch_size = tuple(int(x) for x in self.data_cfg["patch_size"])
        self.base_seed = int(cfg["EXPERIMENT"]["seed"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        volume = int(row.get("volume", _case_volume(row["case_id"])))
        image, label = load_case_volume(
            row["h5_data_dir"],
            volume,
            image_key=self.data_cfg.get("image_key", "image"),
            mask_key=self.data_cfg.get("mask_key", "mask"),
        )
        image = normalize_modalities(image, nonzero=bool(self.data_cfg.get("normalize_nonzero", True)))
        image_low, image_high = wavelet_decompose_3d(image, wavelet_type=self.data_cfg.get("wavelet_type", "haar"))

        rng = np.random.default_rng(self.base_seed + idx + (100000 if self.training else 0))
        if self.training:
            image_low, image_high, label = random_flip_3d(image_low, image_high, label, rng)
            image_low = random_intensity_aug(image_low, rng)
            image_high = random_intensity_aug(image_high, rng)
            image_low, image_high, label = crop_patch_pair(
                image_low,
                image_high,
                label,
                self.patch_size,
                rng,
                foreground_prob=float(self.data_cfg.get("foreground_patch_prob", 0.8)),
            )

        tokenized = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            # Keep image as the low-frequency branch for compatibility with callers.
            "image": torch.from_numpy(image_low).float(),
            "image_low": torch.from_numpy(image_low).float(),
            "image_high": torch.from_numpy(image_high).float(),
            "label": torch.from_numpy(label).long(),
            "text": {
                "input_ids": tokenized["input_ids"].squeeze(0).long(),
                "attention_mask": tokenized["attention_mask"].squeeze(0).long(),
            },
            "case_id": row["case_id"],
            "volume": volume,
            "text_length": int(row.get("token_length", 0)),
        }
