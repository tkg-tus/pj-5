"""
analyze_modalities.py ― モダリティごとの「得意な形」と、モダリティ間の情報の相関を調べる

使い方: python tools/analyze_modalities.py --eval_dir eval_game --system gru
入力:   {eval_dir}/truth.csv    正解（video_id, t, score）。highlight_eval.py labels で作ったもの
        {eval_dir}/preds/{system}_{modality}_seed{n}.csv  予測（seed が複数あれば平均する）
        mosu/mosu_metadata.csv, mosu/vocabulary.csv（タグ別の集計に使う。なければ飛ばす）
出力:   {eval_dir}/analysis_{system}_videos.csv  動画ごとの、正解の形・各モダリティの成績・一番良かったモダリティ
        画面に、次の3つの表を出す
          4. 正解の形（山の数・山の幅・鋭さ）ごとに、どのモダリティが一番良いことが多いか
             （＋タグごと）
          5. モダリティ間の相関：予測どうしの似かた、外れ方（予測 − 正解）どうしの似かた

指標はすべて動画ごとの Spearman（順位の一致）。予測は動画ごとに0〜1にそろえてから使う。
"""
import argparse, ast, csv, glob, os, re
from collections import Counter
import numpy as np
import pandas as pd

MODS = ["visual", "audio", "text"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--system", default="gru")
    p.add_argument("--mosu_dir", default="mosu")
    p.add_argument("--top_tags", type=int, default=8)  # タグ別の表に出すタグの数
    return p.parse_args()


def rank(a):
    return pd.Series(a).rank().to_numpy()


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def normalize(p):
    p = np.asarray(p, float)
    w = p.max() - p.min()
    return (p - p.min()) / w if w > 1e-12 else np.zeros_like(p)


def shape_features(g):
    """正解の曲線の形：山の数、一番高い山の幅、鋭さ、全体の高さ"""
    g = np.asarray(g, float)
    T = len(g)
    above = g >= 0.5  # 最大（1）の半分以上の秒を「山」とみなす
    n_peaks = int(np.sum(above[1:] & ~above[:-1]) + above[0])
    i = int(np.argmax(g))
    lo = i
    while lo > 0 and g[lo - 1] >= 0.5:
        lo -= 1
    hi = i
    while hi < T - 1 and g[hi + 1] >= 0.5:
        hi += 1
    width = hi - lo + 1
    return {"seconds": T, "n_peaks": n_peaks, "peak_width_sec": width, "peak_width_ratio": width / T,
            "peak_pos_ratio": i / max(T - 1, 1), "gt_mean": float(g.mean())}


def load_preds(eval_dir, system):
    preds = {}
    for m in MODS:
        files = sorted(glob.glob(f"{eval_dir}/preds/{system}_{m}_seed*.csv"))
        if not files:
            continue
        per_seed = [pd.read_csv(f, dtype={"video_id": str}) for f in files]
        df = per_seed[0][["video_id", "t"]].copy()
        df["score"] = np.mean([d["score"].to_numpy() for d in per_seed], axis=0)
        preds[m] = {v: normalize(g.sort_values("t")["score"].to_numpy()) for v, g in df.groupby("video_id")}
        print(f"{system} {m}: seed {len(files)}個を平均")
    return preds


def load_tags(mosu_dir):
    meta, voc = f"{mosu_dir}/mosu_metadata.csv", f"{mosu_dir}/vocabulary.csv"
    if not (os.path.exists(meta) and os.path.exists(voc)):
        return None
    with open(voc, encoding="utf-8") as f:
        name = {int(r["Index"]): r["Name"] for r in csv.DictReader(f)}
    with open(meta, encoding="utf-8-sig") as f:
        return {r["video_id"]: [name.get(t, str(t)) for t in ast.literal_eval(r["labels"])] for r in csv.DictReader(f)}


def main():
    args = parse_args()
    truth = pd.read_csv(f"{args.eval_dir}/truth.csv", dtype={"video_id": str})
    gt = {v: g.sort_values("t")["score"].to_numpy() for v, g in truth.groupby("video_id")}
    preds = load_preds(args.eval_dir, args.system)
    mods = [m for m in MODS if m in preds]
    if len(mods) < 2:
        raise SystemExit(f"{args.system} のモダリティが2つ以上見つからない: {mods}")

    rows = []
    for v, g in gt.items():
        if not all(v in preds[m] for m in mods):
            continue
        T = min([len(g)] + [len(preds[m][v]) for m in mods])
        g = g[:T]
        p = {m: preds[m][v][:T] for m in mods}
        row = {"video_id": v} | shape_features(g)
        for m in mods:
            row[f"sp_{m}"] = spearman(p[m], g)
        for a_i, a in enumerate(mods):
            for b in mods[a_i + 1:]:
                row[f"pred_corr_{a}_{b}"] = spearman(p[a], p[b])          # 予測どうしの似かた
                row[f"err_corr_{a}_{b}"] = spearman(p[a] - g, p[b] - g)   # 外れ方どうしの似かた
        sps = {m: row[f"sp_{m}"] for m in mods if not np.isnan(row[f"sp_{m}"])}
        row["best"] = max(sps, key=sps.get) if sps else None
        rows.append(row)
    df = pd.DataFrame(rows)
    out = f"{args.eval_dir}/analysis_{args.system}_videos.csv"
    df.to_csv(out, index=False)
    print(f"\n動画 {len(df)}本を分析 → {out}")

    # ---- 4. 正解の形ごとの、一番良いモダリティ ----
    def win_table(group_col, labels):
        t = df.groupby(group_col, observed=True).agg(
            videos=("video_id", "size"), **{f"{m}_sp": (f"sp_{m}", "mean") for m in mods})
        wins = df.groupby(group_col, observed=True)["best"].value_counts(normalize=True).unstack(fill_value=0)
        for m in mods:
            t[f"{m}_best率"] = wins.get(m, 0)
        t.index = [labels.get(i, i) for i in t.index]
        return t.round(3)

    print("\n===== 4. 全体：各モダリティの Spearman の平均と、一番良かった動画の割合 =====")
    overall = {f"{m}_sp": df[f"sp_{m}"].mean() for m in mods} | {f"{m}_best率": (df["best"] == m).mean() for m in mods}
    print(pd.DataFrame([overall]).round(3).to_string(index=False))

    df["山の数"] = pd.cut(df["n_peaks"], [0, 1, 2, 100], labels=["1つ", "2つ", "3つ以上"])
    print("\n===== 4-1. 正解の山の数ごと =====")
    print(win_table("山の数", {}).to_string())

    df["山の幅"] = pd.qcut(df["peak_width_ratio"], 3, labels=["狭い（鋭い山）", "中くらい", "広い（なだらか）"], duplicates="drop")
    print("\n===== 4-2. 一番高い山の幅（動画の長さに対する割合）ごと =====")
    print(win_table("山の幅", {}).to_string())

    df["山の位置"] = pd.cut(df["peak_pos_ratio"], [-0.01, 0.2, 0.8, 1.01], labels=["序盤（最初の2割）", "中盤", "終盤（最後の2割）"])
    print("\n===== 4-3. 一番高い山の位置ごと =====")
    print(win_table("山の位置", {}).to_string())

    tags = load_tags(args.mosu_dir)
    if tags:
        common = Counter(t for v in df["video_id"] for t in tags.get(v, []) if t not in ("Game", "Video game"))
        print(f"\n===== 4-4. タグごと（多いタグ上位{args.top_tags}個） =====")
        trs = []
        for tag, n in common.most_common(args.top_tags):
            sub = df[df["video_id"].map(lambda v: tag in tags.get(v, []))]
            trs.append({"tag": tag, "videos": len(sub)} | {f"{m}_sp": sub[f"sp_{m}"].mean() for m in mods}
                       | {f"{m}_best率": (sub["best"] == m).mean() for m in mods})
        print(pd.DataFrame(trs).round(3).to_string(index=False))
    else:
        print("\n（タグ別の表は、mosu/mosu_metadata.csv と mosu/vocabulary.csv がないので飛ばした）")

    # ---- 5. モダリティ間の相関 ----
    print("\n===== 5. モダリティ間の相関（動画ごとの Spearman の平均） =====")
    pairs = [(a, b) for i, a in enumerate(mods) for b in mods[i + 1:]]
    print(pd.DataFrame([{"組み合わせ": f"{a} と {b}", "予測どうし": df[f"pred_corr_{a}_{b}"].mean(),
                         "外れ方どうし": df[f"err_corr_{a}_{b}"].mean()} for a, b in pairs]).round(3).to_string(index=False))
    print("\n外れ方どうしの相関が低いほど、統合で外れが打ち消し合いやすい。予測どうしが似ていても、"
          "外れ方が違えば統合は効く。")


if __name__ == "__main__":
    main()
