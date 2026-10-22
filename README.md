# highlight-project

動画を読み込み、1秒ごとの「見どころ度」を予測して、見どころを切り抜くプロジェクト。

## 全体の流れ

```
動画 (videos/*.mp4)
  │  features/extract_features.py   動画 → 1秒ごとの特徴量
  ▼
data/features/{visual,audio,text}/{video_id}.npy   形（秒数, 次元）
  │  train.py                        特徴量 + 正解 → モデルを学習
  ▼
runs/{system}_{modality}_seed{n}/best.pt
  │  highlight/predict.py            新しい動画 → 1秒ごとのスコア
  ▼
スコア（秒数,）
  │  highlight/clip.py               スコア → 見どころ区間 → 動画を切り抜く
  ▼
クリップ (outputs/*.mp4)
  │  frontend/                       結果を見るための画面（未着手）
```

## フォルダ構成

| 場所 | 中身 | 状態 |
|---|---|---|
| `train.py` | 学習スクリプト（共通） | 基本は書き換えない |
| `systems/` | 各自のモデルと学習（1人1ファイル） | `_template.py` をコピーして実装。`systems/README.md` 参照 |
| `features/` | 動画 → 特徴量の変換 | 未実装（雛形のみ） |
| `highlight/` | 学習済みモデルで予測し、切り抜く | 未実装（雛形のみ） |
| `frontend/` | 画面 | 未着手（置き場所だけ） |
| `jobs/` | 計算サーバ（Slurm）に投入するジョブスクリプト | |
| `data/` | 特徴量・正解・分割（Gitには入れない） | `data/README.md` 参照 |
| `runs/` | 学習結果（Gitには入れない） | 実行すると自動で作られる |
| `logs/` | ジョブの出力 `.out`（Gitには入れない） | |

## 環境構築（IS計算サーバ）

ログインノードで実行する（計算ノードには uv が入っていない）。

```bash
cd ~/highlight-project
uv lock
uv sync --locked
```

## 学習の実行

### 動作確認（対話セッション、30分まで）

```bash
srun -p a100_1g --gres=gpu:1g.10gb:1 --qos=inter --pty bash
source .venv/bin/activate
python train.py --system fast --modality visual --seed 0 --subset 5 --max_epochs 2
exit
```

### 本番（バッチジョブ）

プロジェクトのルートで実行する（ジョブは sbatch を実行した場所で始まるため）。

```bash
sbatch -J train_fast_visual \
  -p a100_3g --gres=gpu:3g.40gb:1 \
  --export=ALL,SYSTEM=fast,MODALITY=visual \
  jobs/train.sbatch
```

- `--qos` は所属によって異なる。サーバ利用ガイド 2.2 を確認すること
- 学習時間を比べるときは、全員同じパーティション（GPU）を使う

### ジョブの確認

```bash
squeue -u "$USER"
tail -f logs/<ジョブ名>-<ジョブID>.out
scancel <ジョブID>
```

## 出力

`runs/{system}_{modality}_seed{n}/`

- `best.pt` valの誤差が一番小さかった時点のモデル
- `pred_val.csv` valの予測（video_id, t, score）
- `summary.json` エポックごとの記録、最良の成績、学習時間、パラメータ数、GPU
