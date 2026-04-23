import json
import os
import torch
import pandas as pd
from monai.transforms import (Compose, NormalizeIntensityd, RandRotated, RandZoomd, Resized, ToTensord, LoadImaged, EnsureChannelFirstd, RandGaussianNoised)
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SegData(Dataset):

    def __init__(self, dataname, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224, 224]):
        super(SegData, self).__init__()

        self.dataname = dataname
        self.mode = mode
        self.root_path = root_path
        self.image_size = image_size

        try:
            with open(csv_path, 'r') as f:
                self.data = pd.read_csv(f)
                if dataname == "cov19":
                    self.caption_list = {image: caption for image, caption in zip(self.data['Image'], self.data['Description'])}
                else:
                    self.caption_list = {image: caption for image, caption in zip(self.data['Image'], self.data['text'])}
        except:
            with open(csv_path, 'r') as f:
                self.caption_list = json.load(f)

        self.img_H_name = 'Images_H'
        self.img_L_name = 'Images_L'
        self.prior_folder_name = 'prior_masks'

        target_H_path = os.path.join(self.root_path, self.img_H_name)
        target_L_path = os.path.join(self.root_path, self.img_L_name)
        target_prior_path = os.path.join(self.root_path, self.prior_folder_name)

        if not os.path.exists(target_H_path) or not os.path.exists(target_L_path):
            raise FileNotFoundError(f"Missing {self.img_H_name} or {self.img_L_name} in {self.root_path}")

        if not os.path.exists(target_prior_path):
            raise FileNotFoundError(f"Missing {self.prior_folder_name} in {self.root_path}")

        self.img_folder_name = self.img_H_name
        self.img2_folder_name = self.img_L_name

        all_images = os.listdir(os.path.join(self.root_path, self.img_folder_name))
        self.img_name_map = {os.path.splitext(f)[0]: f for f in all_images}

        self.output_path = os.path.join(self.root_path, 'GTs')
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Missing GTs in {self.root_path}")
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

        print(f"[{mode}] Ready: {len(self.data_pairs)}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

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
        token_output = self.tokenizer.encode_plus(caption, padding='max_length', max_length=24, truncation=True, return_attention_mask=True, return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image': image, 'image2': image2, 'prior': prior, 'gt': gt, 'token': token, 'mask': mask}

        try:
            data = trans(data)
        except Exception as e:
            print(f"Failed for: {image}")
            raise e

        image, image2, prior_mask, gt = data['image'], data['image2'], data['prior'], data['gt']
        token, mask = data['token'], data['mask']

        if gt.shape[0] == 3: gt = gt[0:1, :, :]
        gt = torch.where(gt == 255, 1, 0)

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
                RandZoomd(keys=keys, min_zoom=0.95, max_zoom=1.15, mode=["bicubic", "bicubic", "nearest", "nearest"], prob=0.3),
                RandRotated(keys=keys, range_x=[-0.3, 0.3], keep_size=True, mode=['bicubic', 'bicubic', 'nearest', 'nearest'], prob=0.3),
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