# systems/

各自のモデルと学習の実装を置く場所。1人1ファイル。

## 始め方

```bash
cp systems/_template.py systems/fast.py   # fast は自分のシステム名
```

`systems/fast.py` の `build_model` と `train_one_epoch` を実装して、次で学習する。

```bash
python train.py --system fast --modality visual --seed 0
```

## ルール

- 自分のファイル以外は書き換えない（`train.py` など共通部分を変えるときはチームに相談）
- ファイル名がそのまま `--system` の名前になり、結果は `runs/{system}_{modality}_seed{n}/` に保存される
- 共通の `batches` などは `from train import batches` で使える
