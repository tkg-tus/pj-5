#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

# 動作確認：GPUが見えるか確かめ、動画5本・2エポックだけ学習する
# 投入（プロジェクトのルートで）:
#   sbatch -J smoke -p a100_1g --gres=gpu:1g.10gb:1 --qos=low jobs/smoke.sbatch

set -euo pipefail

echo "ジョブID: $SLURM_JOB_ID"
echo "実行ノード: $(hostname)"
nvidia-smi -L

source .venv/bin/activate
python -u -c "import torch; print('cuda:', torch.cuda.is_available())"
python -u train.py --system "${SYSTEM:-fast}" --modality "${MODALITY:-visual}" --seed 0 \
  --subset 5 --max_epochs 2 --out_dir runs_smoke
