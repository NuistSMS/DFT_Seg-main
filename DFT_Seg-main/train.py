import warnings
import os
import logging
import time
import datetime

# ============== 抑制所有警告 ==============
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置logging级别（可选）
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch

# 强制禁用 weights_only
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'

import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import SegData
import utils.config as config
from torch.optim import lr_scheduler
from net.creratemodel import CreateModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import numpy as np
import random


# ==========================================
# 动态读取早停配置的 ETA 倒计时回调 (终极修复版)
# ==========================================
class ETACallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = 0
        self.epoch_times = []
        # 自己独立维护最优分数和等待轮数，不依赖外部回调的执行顺序
        self.best_miou = -1.0
        self.my_wait_count = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        # 避免在 Sanity Check (初步检查) 阶段打印干扰视线
        if trainer.sanity_checking:
            return

        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)

        avg_epoch_duration = sum(self.epoch_times) / len(self.epoch_times)

        patience = 0
        has_early_stopping = False

        # 遍历 callbacks 只是为了获取你设置的 patience 极限值
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                patience = cb.patience
                has_early_stopping = True
                break

        # 获取刚才存入的各项指标及其方差
        metrics = trainer.callback_metrics
        val_dice = metrics.get('val_dice', torch.tensor(0.0)).item()
        val_dice_var = metrics.get('val_dice_var', torch.tensor(0.0)).item()
        val_miou = metrics.get('val_MIoU', torch.tensor(0.0)).item()
        val_miou_var = metrics.get('val_MIoU_var', torch.tensor(0.0)).item()

        metric_str = f"| Val Dice: {val_dice:.4f} (Var: {val_dice_var:.5f}) | Val MIoU: {val_miou:.4f} (Var: {val_miou_var:.5f})"

        # 核心逻辑：不求人，自己独立判断是否破纪录
        is_best = False
        if val_miou > self.best_miou:
            self.best_miou = val_miou
            self.my_wait_count = 0  # 破纪录，重置计数器
            is_best = True
        else:
            self.my_wait_count += 1  # 没破纪录，计数器 +1

        # 计算理论最大剩余时间
        max_epochs = trainer.max_epochs or 0
        remaining_epochs = max(0, max_epochs - trainer.current_epoch - 1)
        max_eta_seconds = int(avg_epoch_duration * remaining_epochs)
        max_eta = str(datetime.timedelta(seconds=max_eta_seconds))

        if not has_early_stopping:
            print(f"\n[Epoch {trainer.current_epoch}] 耗时: {epoch_duration:.2f}秒 | 预计还需: {max_eta}")
            print(f"  --> {metric_str}")
        else:
            remaining_patience = patience - self.my_wait_count
            early_stop_seconds = int(avg_epoch_duration * remaining_patience)
            early_stop_eta = str(datetime.timedelta(seconds=early_stop_seconds))

            if is_best and trainer.current_epoch > 0:
                print(
                    f"\n[Epoch {trainer.current_epoch}] 耗时: {epoch_duration:.2f}秒 | ⭐ MIoU 破纪录！ | 若后续一直不提升，将在 {early_stop_eta} 后早停退出")
                print(f"  --> {metric_str}")
            else:
                print(
                    f"\n[Epoch {trainer.current_epoch}] 耗时: {epoch_duration:.2f}秒 | ⚠️ 已有 {self.my_wait_count}/{patience} 轮无提升 | 距离早停退出倒计时: {early_stop_eta} (极限最长耗时: {max_eta})")
                print(f"  --> {metric_str}")


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/train.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


if __name__ == '__main__':
    args = get_parser()
    print("cuda is used:", torch.cuda.is_available())

    ds_train = SegData(dataname="cov1",  # cov19
                       csv_path=args.train_csv_path,
                       root_path=args.train_root_path,
                       tokenizer=args.bert_type,
                       image_size=args.image_size,
                       mode='train')

    ds_valid = SegData(dataname="cov1",  # cov19
                       csv_path=args.valid_csv_path,
                       root_path=args.valid_root_path,
                       tokenizer=args.bert_type,
                       image_size=args.image_size,
                       mode='valid')

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

    model = CreateModel(args)

    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_MIoU',
        save_top_k=1,
        mode='max',
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor='val_MIoU',
        patience=args.patience,  # 完全由 args.patience 决定
        mode='max',
    )

    eta_callback = ETACallback()

    trainer = pl.Trainer(logger=True,
                         min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         accelerator='gpu',
                         devices=args.device,
                         callbacks=[model_ckpt, early_stopping, eta_callback],
                         enable_progress_bar=False,
                         )

    print('====start====')
    trainer.fit(model, dl_train, dl_valid)
    print('====finish====')