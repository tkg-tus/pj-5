"""
predict.py ― 学習済みモデルで、新しい動画の1秒ごとのスコアを出す

使い方: python highlight/predict.py --run runs/fast_visual_seed0 --features data/features/visual/{video_id}.npy
入力:   学習済みモデル（best.pt）と、動画1本分の特徴量（秒数, 次元）
出力:   1秒ごとのスコア（秒数,） 0〜1に正規化

★未実装：load_model / predict_scores
"""
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)       # runs/{system}_{modality}_seed{n}
    p.add_argument("--features", required=True)  # 動画1本分の .npy
    p.add_argument("--out", default="scores.npy")
    return p.parse_args()


def load_model(run_dir, in_dim):
    # systems/{system}.py の build_model で同じ形のモデルを作り、best.pt を読み込む
    raise NotImplementedError


def predict_scores(model, x):
    # x:（秒数, 次元） → 各秒のスコア（秒数,） 動画ごとに0〜1へ正規化
    raise NotImplementedError


def main():
    args = parse_args()
    x = np.load(args.features).astype(np.float32)
    model = load_model(args.run, x.shape[1])
    np.save(args.out, predict_scores(model, x))


if __name__ == "__main__":
    main()
