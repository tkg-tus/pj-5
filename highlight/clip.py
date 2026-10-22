"""
clip.py ― スコアから見どころ区間を選び、動画を切り抜く

使い方: python highlight/clip.py --video videos/{video_id}.mp4 --scores scores.npy
入力:   元の動画と、1秒ごとのスコア（秒数,）
出力:   outputs/{video_id}_{n}.mp4  見どころのクリップ

★未実装：select_segments / cut_clip
"""
import argparse, os
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--max_clips", type=int, default=3)
    return p.parse_args()


def select_segments(scores, max_clips):
    # スコアから見どころ区間を選ぶ → [(開始秒, 終了秒), ...]
    raise NotImplementedError


def cut_clip(video_path, start, end, out_path):
    # 動画の start〜end 秒を切り出して保存する（ffmpeg など）
    raise NotImplementedError


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scores = np.load(args.scores)
    vid = Path(args.video).stem
    for n, (start, end) in enumerate(select_segments(scores, args.max_clips)):
        cut_clip(args.video, start, end, f"{args.out_dir}/{vid}_{n}.mp4")


if __name__ == "__main__":
    main()
