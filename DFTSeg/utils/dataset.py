import json
import os
import torch
import pandas as pd
import numpy as np
import pywt
import cv2
from PIL import Image
from tqdm import tqdm
from monai.transforms import (Compose, NormalizeIntensityd, RandRotated, RandZoomd, Resized, ToTensord, LoadImaged,
                              EnsureChannelFirstd, RandGaussianNoised)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class SegData(Dataset):

    def __init__(self, dataname, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224, 224]):
        super(SegData, self).__init__()

        self.dataname = dataname
        self.mode = mode
        self.root_path = root_path
        self.image_size = image_size

        # 1. 读取 CSV/JSON
        try:
            with open(csv_path, 'r') as f:
                self.data = pd.read_csv(f)
                if dataname == "cov19":
                    self.caption_list = {image: caption for image, caption in
                                         zip(self.data['Image'], self.data['Description'])}
                else:
                    self.caption_list = {image: caption for image, caption in
                                         zip(self.data['Image'], self.data['text'])}
        except:
            with open(csv_path, 'r') as f:
                self.caption_list = json.load(f)

        self.img_H_name = 'Images_H'
        self.img_L_name = 'Images_L'

        target_H_path = os.path.join(self.root_path, self.img_H_name)
        target_L_path = os.path.join(self.root_path, self.img_L_name)

        source_img_folder = None
        for possible_name in ['Images', 'images', 'imgs']:
            p = os.path.join(self.root_path, possible_name)
            if os.path.exists(p):
                source_img_folder = p
                break

        # 自动生成高低频
        need_gen_H = not os.path.exists(target_H_path)
        need_gen_L = not os.path.exists(target_L_path)

        if need_gen_H or need_gen_L:
            if source_img_folder is None:
                raise FileNotFoundError(f"[{mode}] 缺少 Images_H 或 Images_L，且无法找到原始图片来生成！")
            print(f"[{mode}] 正在从小波变换生成高频/低频数据...")
            self.generate_wavelet_folders(source_img_folder, target_L_path, target_H_path)

        # 自动生成空间先验掩码
        self.prior_folder_name = 'prior_masks'
        target_prior_path = os.path.join(self.root_path, self.prior_folder_name)
        need_gen_prior = not os.path.exists(target_prior_path)

        if need_gen_prior:
            if source_img_folder is None:
                raise FileNotFoundError(f"[{mode}] 缺少 prior_masks，且找不到原始图片来生成！")
            print(f"[{mode}] 正在通过 OpenCV 自动提取纯净肺部轮廓...")
            self.generate_prior_masks(source_img_folder, target_prior_path)

        self.img_folder_name = self.img_H_name
        self.img2_folder_name = self.img_L_name

        all_images = os.listdir(os.path.join(self.root_path, self.img_folder_name))
        self.img_name_map = {os.path.splitext(f)[0]: f for f in all_images}

        self.output_path = os.path.join(self.root_path, 'GTs')
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"找不到 GTs 文件夹: {self.output_path}")
        raw_mask_list = os.listdir(self.output_path)

        self.data_pairs = []

        for mask_name in raw_mask_list:
            if mask_name not in self.caption_list:
                continue
            if self.dataname == "cov19":
                stem_name = os.path.splitext(mask_name)[0].replace('mask_', '')
            else:
                stem_name = os.path.splitext(mask_name)[0]

            if stem_name in self.img_name_map:
                real_image_name = self.img_name_map[stem_name]
                self.data_pairs.append((mask_name, real_image_name))

        print(f"[{mode}] 数据准备就绪: 共 {len(self.data_pairs)} 组样本")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def generate_wavelet_folders(self, source_path, L_path, H_path, wavelet_type='haar'):
        if not os.path.exists(L_path): os.makedirs(L_path)
        if not os.path.exists(H_path): os.makedirs(H_path)
        file_list = os.listdir(source_path)
        for i in tqdm(file_list, desc="Generating Wavelet Images"):
            try:
                img_path = os.path.join(source_path, i)
                image = Image.open(img_path).convert('L')
                image = np.array(image)
                LL, (LH, HL, HH) = pywt.dwt2(image, wavelet_type)

                LL_norm = (LL - LL.min()) / (LL.max() - LL.min()) * 255
                Image.fromarray(LL_norm.astype(np.uint8)).save(os.path.join(L_path, i))

                merge = HH + HL + LH
                merge_norm = (merge - merge.min()) / (merge.max() - merge.min()) * 255
                Image.fromarray(merge_norm.astype(np.uint8)).save(os.path.join(H_path, i))
            except:
                continue

    def generate_prior_masks(self, source_path, target_path):
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        file_list = os.listdir(source_path)
        for i in tqdm(file_list, desc="Generating Prior Masks (OpenCV)"):
            try:
                img_path = os.path.join(source_path, i)
                out_path = os.path.join(target_path, i)
                if os.path.exists(out_path): continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                blurred = cv2.GaussianBlur(img, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)

                h, w = binary.shape
                mask_ff = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(binary, mask_ff, (0, 0), 0)
                cv2.floodFill(binary, mask_ff, (0, h - 1), 0)
                cv2.floodFill(binary, mask_ff, (w - 1, 0), 0)
                cv2.floodFill(binary, mask_ff, (w - 1, h - 1), 0)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                final_mask = np.zeros_like(binary)
                if num_labels >= 2:
                    sorted_labels = np.argsort(stats[:, 4])[::-1]
                    for label_idx in sorted_labels[1:min(5, len(sorted_labels))]:
                        if stats[label_idx, 4] > 1000:
                            final_mask[labels == label_idx] = 255
                else:
                    final_mask = binary

                kernel = np.ones((15, 15), np.uint8)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
                cv2.imwrite(out_path, final_mask)
            except:
                continue

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        mask_name, real_image_name = self.data_pairs[idx]
        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path, self.img_folder_name, real_image_name)
        image2 = os.path.join(self.root_path, self.img2_folder_name, real_image_name)
        prior = os.path.join(self.root_path, self.prior_folder_name, real_image_name)
        gt = os.path.join(self.root_path, 'GTs', mask_name)

        caption = self.caption_list[mask_name]
        token_output = self.tokenizer.encode_plus(caption, padding='max_length', max_length=24,
                                                  truncation=True, return_attention_mask=True, return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        # 包含 prior 在内的数据字典
        data = {'image': image, 'image2': image2, 'prior': prior, 'gt': gt, 'token': token, 'mask': mask}

        try:
            data = trans(data)
        except Exception as e:
            print(f"[Error] Loading failed for: {image}")
            raise e

        image, image2, prior_mask, gt = data['image'], data['image2'], data['prior'], data['gt']
        token, mask = data['token'], data['mask']

        if gt.shape[0] == 3: gt = gt[0:1, :, :]
        gt = torch.where(gt == 255, 1, 0)

        # 处理 prior_mask 为 0~1 的浮点张量
        if prior_mask.shape[0] == 3: prior_mask = prior_mask[0:1, :, :]
        if prior_mask.max() > 1.0: prior_mask = prior_mask / 255.0

        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)}

        return ([image, image2, text, prior_mask], gt)

    def transform(self, image_size=[224, 224]):
        keys = ["image", "image2", "prior", "gt"]
        if self.mode == 'train':
            trans = Compose([
                LoadImaged(keys=keys, reader='PILReader'),
                EnsureChannelFirstd(keys=keys),
                RandGaussianNoised(keys=["image2"], prob=0.3, mean=0.0, std=0.1),
                RandZoomd(keys=keys, min_zoom=0.95, max_zoom=1.15, mode=["bicubic", "bicubic", "nearest", "nearest"],
                          prob=0.3),
                RandRotated(keys=keys, range_x=[-0.3, 0.3], keep_size=True,
                            mode=['bicubic', 'bicubic', 'nearest', 'nearest'], prob=0.3),
                Resized(["image", "image2"], spatial_size=image_size, mode='bicubic'),
                Resized(["prior", "gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(keys=["image", "image2"], channel_wise=True),
                ToTensord(keys=["image", "image2", "prior", "gt", "token", "mask"])
            ])
        else:
            trans = Compose([
                LoadImaged(keys=keys, reader='PILReader'),
                EnsureChannelFirstd(keys=keys),
                Resized(["image", "image2"], spatial_size=image_size, mode='bicubic'),
                Resized(["prior", "gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(keys=["image", "image2"], channel_wise=True),
                ToTensord(keys=["image", "image2", "prior", "gt", "token", "mask"]),
            ])
        return trans