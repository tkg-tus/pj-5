"""
systems/_template.py ― 各自の実装の雛形

使い方: このファイルを systems/{自分のシステム名}.py にコピーして、2つの関数を実装する
        python train.py --system {自分のシステム名} --modality visual --seed 0
"""
import torch
import torch.nn as nn

from train import batches


# ★各自で実装：モデル
def build_model(in_dim):
    # forward(X, M) →（バッチ, 秒数）の各秒のスコア を返す nn.Module
    raise NotImplementedError


# ★各自で実装：1エポック分の学習
def train_one_epoch(model, data, opt, args, dev, rng):
    # batches(data, args.batch_size, dev, rng) でシャッフルしたバッチが取れる
    # 詰め物の部分（M が False）は損失に含めない
    # 戻り値：平均の学習損失
    raise NotImplementedError
