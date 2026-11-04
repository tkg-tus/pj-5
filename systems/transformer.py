"""
systems/transformer.py ― Transformer Encoder（自己注意で動画全体を見る）

使い方: python train.py --system transformer --modality visual --seed 0
自分のモデルは systems/{自分のシステム名}.py に作る（このファイルは EI 担当）
"""
import math

import torch
import torch.nn as nn

from train import batches

D_MODEL = 128  # Transformer の幅
N_HEAD = 4  # 注意のヘッド数（D_MODEL を割り切れる数）
N_LAYERS = 2  # 層の数
D_FF = 256  # 各層の全結合の幅
DROPOUT = 0.1
CLIP = 1.0  # 勾配クリッピングの上限（lr=1e-3 固定でも暴れにくくする）


def positional_encoding(T, d, dev):
    """sin/cos の位置エンコーディング（T, d）。秒数に合わせて毎回作るので最大長を決めなくてよい"""
    pos = torch.arange(T, device=dev, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=dev, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=dev)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class HighlightTransformer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_FF, DROPOUT,
                                           batch_first=True, norm_first=True)  # Pre-LN：学習が安定しやすい
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)  # Pre-LN は最後の出力が正規化されないので、ここで正規化
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, X, M):
        # X:（バッチ, 秒数, 次元） M:（バッチ, 秒数） 有効な秒が True
        h = self.inp(X) * math.sqrt(D_MODEL) + positional_encoding(X.shape[1], D_MODEL, X.device)
        h = self.encoder(self.drop(h), src_key_padding_mask=~M)  # 詰め物（False）は注意の対象から外す
        return self.head(self.norm(h)).squeeze(-1)  # （バッチ, 秒数）のスコア


def build_model(in_dim):
    return HighlightTransformer(in_dim)


def train_one_epoch(model, data, opt, args, dev, rng):
    model.train()
    total, n = 0.0, 0
    for _, X, Y, M in batches(data, args.batch_size, dev, rng):
        P = model(X, M)
        loss = ((P - Y) ** 2)[M].mean()  # 詰め物の秒は除いて平均二乗誤差
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        total += loss.item(); n += 1
    return total / n
