#!/usr/bin/env python3
"""MoSu等の1秒単位の見どころスコアを評価・比較・統合するCLI.

CSV仕様: video_id,t,score  (tは0始まりの整数秒、scoreは0..1)

使い方: python highlight_eval.py --help
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KEYS = ["video_id", "t"]
COLS = KEYS + ["score"]
METRICS = ["spearman", "mae", "top10_overlap"]


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"保存: {path} ({len(df)}行)")


def read_scores(path: str | Path, require_contiguous: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"video_id": "string"})
    if not set(COLS).issubset(df.columns):
        raise ValueError(f"{path}: 必須列 {COLS} がありません")
    df = df[COLS].copy()
    if df["video_id"].isna().any() or (df["video_id"].str.strip() == "").any():
        raise ValueError(f"{path}: 空のvideo_idがあります")
    for col in ("t", "score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["t", "score"]].isna().any().any():
        raise ValueError(f"{path}: 時刻またはスコアに欠損・文字列があります")
    if not np.isfinite(df[["t", "score"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: 無限大などの不正値があります")
    if (df["t"] < 0).any() or (df["t"] % 1 != 0).any():
        raise ValueError(f"{path}: tは0以上の整数秒にしてください")
    df["t"] = df["t"].astype(np.int64)
    if ((df["score"] < 0) | (df["score"] > 1)).any():
        raise ValueError(f"{path}: scoreは0～1の範囲にしてください")
    if df.duplicated(KEYS).any():
        raise ValueError(f"{path}: (video_id,t)が重複しています")
    df = df.sort_values(KEYS).reset_index(drop=True)
    if require_contiguous:
        for video_id, group in df.groupby("video_id", sort=False):
            times = group["t"].to_numpy()
            if not np.array_equal(times, np.arange(len(times))):
                raise ValueError(f"{path}: {video_id} のtが0秒始まりの連続値ではありません")
    if df.empty:
        raise ValueError(f"{path}: 空のCSVです")
    return df


def align_strict(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> pd.DataFrame:
    aligned = reference[KEYS].merge(candidate, on=KEYS, how="outer", indicator=True, validate="one_to_one")
    missing = int((aligned["_merge"] == "left_only").sum())
    extra = int((aligned["_merge"] == "right_only").sum())
    if missing or extra:
        raise ValueError(f"{name}: 時間軸が不一致 (欠落{missing}行・余分{extra}行)。動画ID・秒数・データ分割を確認")
    return aligned.drop(columns="_merge").sort_values(KEYS).reset_index(drop=True)


def minmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    width = float(np.max(x) - np.min(x))
    return np.zeros_like(x) if width <= 1e-12 else (x - x.min()) / width


def normalize_per_video(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df.groupby("video_id", sort=False)["score"].transform(
        lambda s: pd.Series(minmax(s.to_numpy()), index=s.index)
    )
    return df


def top_fraction_membership(x: np.ndarray, k: int) -> np.ndarray:
    """同点が閾値をまたぐ場合、該当時刻に均等な選択確率を割り振る。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(-x, kind="stable")
    threshold = x[order[k - 1]]
    above = x > threshold
    tied = x == threshold
    out = above.astype(float)
    out[tied] = (k - above.sum()) / tied.sum()
    return out


def calculate_metrics(truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    aligned = align_strict(truth, prediction, "prediction")
    merged = truth.merge(aligned, on=KEYS, suffixes=("_true", "_pred"), validate="one_to_one")
    rows = []
    for video_id, g in merged.groupby("video_id", sort=False):
        y = g["score_true"].to_numpy(dtype=float)
        p = g["score_pred"].to_numpy(dtype=float)
        n = len(y)
        rho = np.nan
        if n >= 2 and np.ptp(y) > 1e-12 and np.ptp(p) > 1e-12:
            rho = float(spearmanr(y, p).statistic)
        k = max(1, math.ceil(n * 0.10))
        top10 = np.nan
        if np.ptp(y) > 1e-12 and np.ptp(p) > 1e-12:
            # タイブレークによる時刻順への依存を回避
            a = top_fraction_membership(y, k)
            b = top_fraction_membership(p, k)
            top10 = float(np.dot(a, b) / k)
        rows.append({
            "video_id": video_id, "seconds": n,
            "spearman": rho, "mae": float(np.mean(np.abs(y - p))),
            "top10_overlap": top10,
            "pred_mean": float(np.mean(p)), "pred_std": float(np.std(p)),
            "pred_peak_t": int(g.iloc[int(np.argmax(p))]["t"]),
            "truth_peak_t": int(g.iloc[int(np.argmax(y))]["t"]),
        })
    return pd.DataFrame(rows)


def summary_row(model: str, metrics: pd.DataFrame) -> dict:
    row = {"model": model, "videos": len(metrics), "seconds": int(metrics["seconds"].sum())}
    for col in METRICS:
        row[col] = float(metrics[col].mean()) if metrics[col].notna().any() else np.nan
        row[col + "_valid_videos"] = int(metrics[col].notna().sum())
    row["mean_pred_std"] = float(metrics["pred_std"].mean())
    row["mean_peak_error_sec"] = float((metrics["pred_peak_t"] - metrics["truth_peak_t"]).abs().mean())
    return row


def load_predictions(paths: list[str], reference: pd.DataFrame | None = None) -> tuple[list[str], list[pd.DataFrame]]:
    if not paths:
        raise ValueError("予測CSVを1個以上指定してください")
    names, frames = [], []
    for path in paths:
        name = Path(path).stem
        if name in names:
            raise ValueError(f"ファイル名（拡張子除く）が重複: {name}")
        df = read_scores(path)
        if reference is not None:
            df = align_strict(reference, df, path)
        names.append(name)
        frames.append(df)
    return names, frames


def make_labels(args: argparse.Namespace) -> None:
    import h5py

    metadata = pd.read_csv(args.metadata, dtype={"video_id": "string"})
    if not {"video_id", "duration"}.issubset(metadata.columns):
        raise ValueError("metadataにはvideo_id,durationが必要です")
    if metadata["video_id"].duplicated().any():
        raise ValueError("metadataのvideo_idが重複しています")
    metadata = metadata.set_index("video_id")
    ids = [line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("idsファイルに重複IDがあります")
    rows, skipped = [], []
    with h5py.File(args.gt, "r") as h5:
        for vid in ids:
            if vid not in h5 or vid not in metadata.index:
                skipped.append((vid, "gtまたはmetadataなし"))
                continue
            duration = float(metadata.loc[vid, "duration"])
            scores = np.asarray(h5[vid]["gt_score"][()], dtype=float).reshape(-1)
            if not np.isfinite(duration) or duration <= 0 or len(scores) == 0 or not np.isfinite(scores).all():
                skipped.append((vid, "不正なdurationまたはgt_score"))
                continue
            n = math.ceil(duration)
            if args.gt_mode == "per_second":
                # MoSu の gt_score は特徴量と同じ長さで、metadata の duration（整数）より1秒長いことがある
                if len(scores) not in (n, n + 1):
                    raise ValueError(f"{vid}: gt_score長={len(scores)}, ceil(duration)={n}。1秒単位ではない可能性。--gt-mode uniformを使う前に仕様確認")
                second_scores = scores
            else:
                # gt_scoreの各要素が動画全体を均等な時間幅で覆うという明示的な仮定
                edges = np.linspace(0, duration, len(scores) + 1)
                second_scores = []
                for t in range(n):
                    start, end = float(t), min(float(t + 1), duration)
                    overlap = np.maximum(0, np.minimum(edges[1:], end) - np.maximum(edges[:-1], start))
                    second_scores.append(float(np.dot(overlap, scores) / (end - start)))
                second_scores = np.asarray(second_scores)
            if args.normalize == "per_video":
                second_scores = minmax(second_scores)
            elif np.min(second_scores) < 0 or np.max(second_scores) > 1:
                raise ValueError(f"{vid}: gt_scoreが0～1の範囲外です。--normalize per_videoを検討")
            rows.extend({"video_id": vid, "t": t, "score": float(s)} for t, s in enumerate(second_scores))
    if not rows:
        raise ValueError("正解ラベルを作成できませんでした")
    save_csv(pd.DataFrame(rows), args.output)
    if skipped:
        print(f"警告: {len(skipped)}本をスキップ。先頭10件: {skipped[:10]}", file=sys.stderr)


def evaluate(args: argparse.Namespace) -> None:
    truth = read_scores(args.truth)
    pred = read_scores(args.pred)
    metrics = calculate_metrics(truth, pred)
    save_csv(metrics, args.output)
    print(pd.DataFrame([summary_row(Path(args.pred).stem, metrics)]).to_string(index=False))


def compare(args: argparse.Namespace) -> None:
    truth = read_scores(args.truth)
    paths = sorted(str(p) for p in Path(args.pred_dir).glob("*.csv"))
    if not paths:
        raise ValueError("指定ディレクトリにCSVがありません")
    names, predictions = load_predictions(paths, truth)
    summaries, details = [], []
    for name, pred in zip(names, predictions):
        metrics = calculate_metrics(truth, pred)
        summaries.append(summary_row(name, metrics))
        details.append(metrics.assign(model=name))
    df = pd.DataFrame(summaries).sort_values("spearman", ascending=False, na_position="last")
    save_csv(df, args.output)
    if args.details:
        save_csv(pd.concat(details, ignore_index=True), args.details)
    print(df.to_string(index=False))


def merge_predictions(predictions: list[pd.DataFrame], weights: np.ndarray, normalization: str) -> pd.DataFrame:
    reference = predictions[0]
    arrays = []
    for i, pred in enumerate(predictions):
        aligned = align_strict(reference, pred, f"予測{i + 1}")
        if normalization == "per_video":
            aligned = normalize_per_video(aligned)
        arrays.append(aligned["score"].to_numpy(dtype=float))
    matrix = np.column_stack(arrays)
    out = reference[KEYS].copy()
    out["score"] = matrix @ weights
    return out


def ensemble(args: argparse.Namespace) -> None:
    names, preds = load_predictions(args.preds)
    if args.weights is None:
        weights = np.ones(len(preds), dtype=float)
    else:
        weights = np.asarray(args.weights, dtype=float)
    if len(weights) != len(preds) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("重みは予測数と同じ個数の有限・非負の数値で、合計が正である必要があります")
    weights = weights / weights.sum()
    out = merge_predictions(preds, weights, args.normalize_inputs)
    if args.normalize_output == "per_video":
        out = normalize_per_video(out)
    save_csv(out, args.output)
    print("統合重み:", dict(zip(names, weights.round(4))))
    if args.truth:
        metrics = calculate_metrics(read_scores(args.truth), out)
        print("統合後の評価:")
        print(pd.DataFrame([summary_row("ensemble", metrics)]).to_string(index=False))
        if args.details:
            save_csv(metrics, args.details)


def clips(args: argparse.Namespace) -> None:
    if args.length <= 0 or args.top <= 0:
        raise ValueError("--lengthと--topは1以上の整数です")
    pred = read_scores(args.pred)
    rows = []
    for vid, g in pred.groupby("video_id", sort=False):
        score = g["score"].to_numpy(dtype=float)
        if len(score) < args.length:
            continue
        means = np.convolve(score, np.ones(args.length) / args.length, mode="valid")
        # 重複区間なし。高スコア順、同点なら開始が早い順
        used = np.zeros(len(score), dtype=bool)
        selected = 0
        for start in np.argsort(-means, kind="stable"):
            end = int(start) + args.length
            if used[start:end].any():
                continue
            used[start:end] = True
            selected += 1
            rows.append({"video_id": vid, "rank": selected, "start_sec": int(start),
                         "end_sec_exclusive": end, "mean_score": float(means[start])})
            if selected >= args.top:
                break
    if not rows:
        raise ValueError("対象動画が指定区間長より短く、切り抜きを作成できません")
    save_csv(pd.DataFrame(rows), args.output)


def plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = read_scores(args.truth)
    names, preds = load_predictions(args.preds, truth)
    if args.video_id not in set(truth["video_id"]):
        raise ValueError(f"動画ID {args.video_id} は正解CSVにありません")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    gt = truth.loc[truth["video_id"] == args.video_id]
    ax.plot(gt["t"], gt["score"], label="Ground truth", linewidth=2.5)
    for name, pred in zip(names, preds):
        g = pred.loc[pred["video_id"] == args.video_id]
        ax.plot(g["t"], g["score"], label=name, alpha=0.8)
    ax.set(xlabel="Time (seconds)", ylabel="Highlight score", ylim=(-0.03, 1.03),
           title=f"Video: {args.video_id}")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"保存: {args.output}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("labels", help="MoSu gt_scoreから正解CSVを作成")
    q.add_argument("--gt", required=True, help="HDF5ファイル")
    q.add_argument("--metadata", required=True, help="video_id,durationを持つCSV")
    q.add_argument("--ids", required=True, help="対象video_idの改行区切りファイル")
    q.add_argument("--gt-mode", choices=["per_second", "uniform"], default="per_second",
                   help="既に1秒ごとの配列か、動画時間全体を均等分割した配列か（要仕様確認）")
    q.add_argument("--normalize", choices=["per_video", "none"], default="per_video")
    q.add_argument("--output", required=True)
    q.set_defaults(func=make_labels)
    q = sub.add_parser("evaluate", help="単一モデルを評価")
    q.add_argument("--truth", required=True)
    q.add_argument("--pred", required=True)
    q.add_argument("--output", required=True)
    q.set_defaults(func=evaluate)
    q = sub.add_parser("compare", help="フォルダ内の全モデルを同じ正解データで比較")
    q.add_argument("--truth", required=True)
    q.add_argument("--pred-dir", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--details", help="モデル・動画別評価CSV")
    q.set_defaults(func=compare)
    q = sub.add_parser("ensemble", help="複数モデルを加重平均で統合")
    q.add_argument("--preds", nargs="+", required=True)
    q.add_argument("--weights", nargs="+", type=float)
    q.add_argument("--normalize-inputs", choices=["none", "per_video"], default="none")
    q.add_argument("--normalize-output", choices=["none", "per_video"], default="none")
    q.add_argument("--truth", help="指定すれば統合後の評価も表示")
    q.add_argument("--details", help="--truth指定時の動画別評価CSV")
    q.add_argument("--output", required=True)
    q.set_defaults(func=ensemble)
    q = sub.add_parser("clips", help="重複しない高スコア区間を選択")
    q.add_argument("--pred", required=True)
    q.add_argument("--length", type=int, default=30)
    q.add_argument("--top", type=int, default=3)
    q.add_argument("--output", required=True)
    q.set_defaults(func=clips)
    q = sub.add_parser("plot", help="1動画の正解・予測スコアを重ねて描画")
    q.add_argument("--truth", required=True)
    q.add_argument("--preds", nargs="+", required=True)
    q.add_argument("--video-id", required=True)
    q.add_argument("--output", required=True)
    q.set_defaults(func=plot)
    return p


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
