from net.model import DFTSeg
from monai.losses import DiceCELoss
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime


class CreateModel(pl.LightningModule):
    def __init__(self, args):
        super(CreateModel, self).__init__()
        # 实例化改名后的主网络 DFTSeg
        self.model = DFTSeg(args.bert_type, args.vision_type, args.project_dim)
        self.lr = args.lr
        self.history = {}

        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc": Accuracy(task='binary'), "dice": Dice(), "MIoU": BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        # [新增]：用于临时存储每个 batch 的指标，用于在 epoch 结束时算方差
        self.batch_metrics = {"train": {"dice": [], "MIoU": []},
                              "val": {"dice": [], "MIoU": []},
                              "test": {"dice": [], "MIoU": []}}

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x, train_mask=None):
        return self.model.forward(x, train_mask)

    def shared_step(self, batch, batch_idx):
        x, y = batch

        if self.training:
            preds, preds2, y_aug = self(x, train_mask=y)
        else:
            preds, preds2, _ = self(x, train_mask=None)
            y_aug = y

        loss1 = self.loss_fn(preds, y_aug)
        loss2 = self.loss_fn(preds2, y_aug)
        loss = loss1 + loss2

        if isinstance(y_aug, torch.Tensor) and y_aug.is_floating_point():
            y_metric = (y_aug > 0.5).int()
        else:
            y_metric = y_aug.int()

        return {'loss': loss, 'preds': preds.detach(), 'y': y_metric.detach()}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            return self(batch[0])[0]
        else:
            return self(batch)[0]

    def shared_step_end(self, outputs, stage):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage == "train":
                self.log(name, step_metric, prog_bar=True)


            if name in ["dice", "MIoU"]:
                self.batch_metrics[stage][name].append(step_metric)

        return outputs["loss"].mean()

    def training_step_end(self, outputs):
        return {'loss': self.shared_step_end(outputs, "train")}

    def validation_step_end(self, outputs):
        return {'val_loss': self.shared_step_end(outputs, "val")}

    def test_step_end(self, outputs):
        return {'test_loss': self.shared_step_end(outputs, "test")}

    def shared_epoch_end(self, outputs, stage="train"):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)

        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage + "_loss").replace('train_', '')] for t in outputs])).item()
        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for name in metrics:
            epoch_metric = metrics[name].compute().item()
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric


            if name in ["dice", "MIoU"]:
                batch_scores = self.batch_metrics[stage][name]
                variance = np.var(batch_scores) if len(batch_scores) > 0 else 0.0
                dic[stage + "_" + name + "_var"] = variance
                self.batch_metrics[stage][name] = []

        if stage != 'test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic

    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor
        mode = ckpt_cb.mode
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor, arr_scores[best_score_idx]), file=sys.stderr)

    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        dic.pop("epoch", None)
        self.print(dic)
        self.log_dict(dic, logger=True)

    def get_history(self):
        return pd.DataFrame(self.history.values())

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n" + "=" * 80 + "%s" % nowtime)