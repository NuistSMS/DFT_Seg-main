import argparse
from net.creratemodel import CreateModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils.dataset import SegData
import utils.config as config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parser():
    parser = argparse.ArgumentParser(description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config', default='./config/train.yaml', type=str, help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


if __name__ == '__main__':
    args = get_parser()

    # load model
    model = CreateModel(args)
    checkpoint = torch.load('save_model/medseg.ckpt', map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=False)

    # dataloader
    ds_test = SegData(dataname="cov9",
                      csv_path=args.test_csv_path,
                      root_path=args.test_root_path,
                      tokenizer=args.bert_type,
                      image_size=args.image_size,
                      mode='test')

    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    trainer = pl.Trainer(accelerator='gpu', devices=1)
    model.eval()

    # 获取测试结果并打印均值与方差
    results = trainer.test(model, dl_test)

    if results:
        res = results[0]
        print("\n" + "=" * 50)
        print("                FINAL TEST RESULTS                ")
        print("=" * 50)
        print(f" Test Dice : {res.get('test_dice', 0):.4f} ")
        print(f" Test MIoU : {res.get('test_MIoU', 0):.4f} ")
        print("=" * 50 + "\n")