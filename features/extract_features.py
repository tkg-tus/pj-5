"""
extract_features.py ― 動画から1秒ごとの特徴量を作る

使い方: python features/extract_features.py --modality visual --video_dir videos
入力:   {video_dir}/{video_id}.mp4
出力:   data/features/{modality}/{video_id}.npy  形（秒数, 次元）

★未実装：extract_visual / extract_audio / extract_text
"""
import argparse, os
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", required=True, choices=["visual", "audio", "text"])
    p.add_argument("--video_dir", default="videos")
    p.add_argument("--data_dir", default="data")
    return p.parse_args()


def extract_visual(video_path):
    # 1秒ごとにフレームを取り出し、画像モデルで特徴にする →（秒数, 次元）
    raise NotImplementedError


def extract_audio(video_path):
    # 音声を取り出し、1秒ごとに音のモデルで特徴にする →（秒数, 次元）
    raise NotImplementedError


def extract_text(video_path):
    # 字幕や文字起こしを1秒ごとに割り当て、文のモデルで特徴にする →（秒数, 次元）
    raise NotImplementedError


EXTRACTORS = {"visual": extract_visual, "audio": extract_audio, "text": extract_text}


def main():
    args = parse_args()
    out = Path(args.data_dir) / "features" / args.modality
    os.makedirs(out, exist_ok=True)
    for video in sorted(Path(args.video_dir).glob("*.mp4")):
        x = EXTRACTORS[args.modality](video).astype(np.float32)
        np.save(out / f"{video.stem}.npy", x)
        print(video.stem, x.shape)


if __name__ == "__main__":
    main()
