from net.model import SegModel
from monai.losses import DiceCELoss
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
try:
    from medpy.metric.binary import hd95
except ImportError:
    hd95 = None


class BinaryScoreMetric(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.scores = []

    def forward(self, preds, target):
        preds = (preds > 0.5).float().view(-1)
        target = (target > 0.5).float().view(-1)

        if self.mode == "acc":
            score = (preds == target).float().mean()
        elif self.mode == "dice":
            intersection = (preds * target).sum()
            denom = preds.sum() + target.sum()
            score = (2.0 * intersection + 1e-6) / (denom + 1e-6)
        else:
            raise ValueError(f"Unsupported metric mode: {self.mode}")

        self.scores.append(score.detach())
        return score

    def compute(self):
        if len(self.scores) == 0:
            return torch.tensor(0.0)
        return torch.stack(self.scores).mean()

    def reset(self):
        self.scores = []


# 最纯净版的 HD95 评估模块，尊重模型原始输出
class HD95Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = []

    def forward(self, preds, target):
        if hd95 is None:
            return torch.tensor(0.0, device=preds.device)

        preds_np = (preds > 0.5).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        batch_scores = []
        for i in range(preds_np.shape[0]):
            p = preds_np[i]
            t = target_np[i]

            # 不做任何后处理，直接计算最真实的 HD95
            if np.count_nonzero(p) > 0 and np.count_nonzero(t) > 0:
                score = hd95(p, t)
            elif np.count_nonzero(p) == 0 and np.count_nonzero(t) == 0:
                score = 0.0  # 完美匹配全黑背景
            else:
                score = 50.0  # 预测全错的惩罚上限

            batch_scores.append(score)

        if len(batch_scores) > 0:
            mean_score = np.mean(batch_scores)
            self.scores.append(mean_score)
            return torch.tensor(mean_score)
        else:
            return torch.tensor(0.0)

    def compute(self):
        return torch.tensor(np.mean(self.scores) if len(self.scores) > 0 else 0.0)

    def reset(self):
        self.scores = []


class CreateModel(pl.LightningModule):
    def __init__(self, args):
        super(CreateModel, self).__init__()
        self.model = SegModel(args.bert_type, args.vision_type, args.project_dim,
                              max_text_len=getattr(args, 'max_text_len', 24))
        self.lr = args.lr
        self.history = {}

        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc": BinaryScoreMetric("acc"), "dice": BinaryScoreMetric("dice"),
                        "MIoU": BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        # 仅给测试集添加 HD95 评估模块
        self.test_metrics["HD95"] = HD95Metric()

        # 用于临时存储每个 batch 的指标，用于在 epoch 结束时算方差
        self.batch_metrics = {"train": {"dice": [], "MIoU": []},
                              "val": {"dice": [], "MIoU": []},
                              "test": {"dice": [], "MIoU": [], "HD95": []}}
        self.batch_losses = {"train": [], "val": [], "test": []}

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
        outputs = self.shared_step(batch, batch_idx)
        loss = self.shared_step_end(outputs, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        return self.shared_step_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        return self.shared_step_end(outputs, "test")

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

            if name in ["dice", "MIoU", "HD95"]:
                self.batch_metrics[stage][name].append(step_metric)

        loss = outputs["loss"].mean()
        self.batch_losses[stage].append(loss.detach())
        return loss

    def _legacy_training_step_end(self, outputs):
        return {'loss': self.shared_step_end(outputs, "train")}

    def _legacy_validation_step_end(self, outputs):
        return {'val_loss': self.shared_step_end(outputs, "val")}

    def _legacy_test_step_end(self, outputs):
        return {'test_loss': self.shared_step_end(outputs, "test")}

    def shared_epoch_end(self, stage="train"):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)

        epoch = self.trainer.current_epoch
        losses = self.batch_losses[stage]
        stage_loss = torch.stack(losses).mean().item() if len(losses) > 0 else 0.0
        self.batch_losses[stage] = []
        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for name in metrics:
            epoch_metric = metrics[name].compute().item()
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric

            if name in ["dice", "MIoU", "HD95"]:
                batch_scores = self.batch_metrics[stage][name]
                variance = np.var(batch_scores) if len(batch_scores) > 0 else 0.0
                dic[stage + "_" + name + "_var"] = variance
                self.batch_metrics[stage][name] = []

        if stage != 'test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic

    def on_train_epoch_end(self):
        dic = self.shared_epoch_end(stage="train")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def on_validation_epoch_end(self):
        dic = self.shared_epoch_end(stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

        ckpt_cb = self.trainer.checkpoint_callback
        if ckpt_cb is None:
            return
        monitor = ckpt_cb.monitor
        mode = ckpt_cb.mode
        history = self.get_history()
        if monitor not in history:
            return
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor, arr_scores[best_score_idx]), file=sys.stderr)

    def on_test_epoch_end(self):
        dic = self.shared_epoch_end(stage="test")
        dic.pop("epoch", None)
        self.print(dic)
        self.log_dict(dic, logger=True)

    def get_history(self):
        return pd.DataFrame(self.history.values())

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n" + "=" * 80 + "%s" % nowtime)
