"""
make_questions.py ― 人手評価の問題ファイル（questions.json）を作る

使い方: python tools/make_questions.py --videos ja_game_videos.csv --splits val --set_id pilot --n 10
        python tools/make_questions.py --videos ja_game_videos_genre.csv --splits test --genres FPS 格闘 --set_id eval
        python tools/make_questions.py --videos ja_game_videos.csv --splits train --set_id tune
        python tools/make_questions.py --videos ja_game_videos.csv --splits test --set_id eval
入力:   --videos の CSV（find_ja_videos.py の ja_game_videos.csv など。video_id, youtube_id の列が必要）
          use 列があれば use=1 の行だけ使う（Excel で手で印を付けた場合）。
          なければ maybe_not_play=1 の行を除く。genre 列があればそれを、なければ genre_tags の最初のタグをジャンルにする
        mosu/mosu_gt.h5（正解の gt_score）
出力:   questions_{set_id}.json（回答ページ human_eval.html で読み込む）
        画面に、使った本数・使えなかった理由・ジャンルとチャンネルの内訳

候補（1問に3つ、各 --clip 秒）:
  MR1 = 区間内の gt_score の平均が一番高い区間（同じ値なら開始が早い方）
  MR2 = MR1 から --gap 秒以上空いた区間の中で、平均が一番高い区間
  R   = MR1・MR2 のどちらからも --gap 秒以上空いた区間から、乱数で選ぶ
  3つのどの組も、区間と区間の間が --gap 秒以上空く。取れない動画は使わない。
"""
import argparse, csv, json, random
from collections import Counter
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--mosu_dir", default="mosu")
    p.add_argument("--splits", nargs="*", default=None)  # 使う split（例：val）。指定なしなら全部
    p.add_argument("--genres", nargs="*", default=None)  # 使うジャンル（genre 列の値。例：FPS 格闘）。指定なしなら全部
    p.add_argument("--set_id", required=True)  # 問題セットの名前（pilot, tune, eval など）
    p.add_argument("--clip", type=int, default=15)  # 候補の長さ（秒）
    p.add_argument("--gap", type=int, default=30)  # 候補どうしの間の最小の空き（秒）
    p.add_argument("--n", type=int, default=0)  # 問題の数の上限。0なら使える動画を全部
    p.add_argument("--max_per_channel", type=int, default=0)  # 1チャンネルあたりの上限。0なら制限なし
    p.add_argument("--seed", type=int, default=0)  # R の位置と、--n で選ぶ動画を決める乱数（同じ値なら同じ問題になる）
    p.add_argument("--out", default=None)
    return p.parse_args()


def window_means(g, clip):
    """開始秒 s ごとの、s〜s+clip の平均。最後の1秒は動画の長さのずれ（±1秒）に備えて使わない"""
    c = np.concatenate([[0.0], np.cumsum(g)])
    last = len(g) - clip - 1
    if last < 0:
        return np.array([])
    s = np.arange(last + 1)
    return (c[s + clip] - c[s]) / clip


def far_from(starts, others, clip, gap):
    """starts のうち、others のどの区間とも gap 秒以上空いているもの"""
    ok = np.ones(len(starts), bool)
    for o in others:
        ok &= np.abs(starts - o) >= clip + gap
    return ok


def candidates(g, clip, gap, rng):
    m = window_means(np.asarray(g, float), clip)
    if len(m) == 0:
        return None, "短すぎる"
    starts = np.arange(len(m))
    s1 = int(np.argmax(m))
    ok2 = far_from(starts, [s1], clip, gap)
    if not ok2.any():
        return None, "MR2 が取れない"
    s2 = int(starts[ok2][np.argmax(m[ok2])])
    okr = far_from(starts, [s1, s2], clip, gap)
    if not okr.any():
        return None, "R が取れない"
    sr = int(rng.choice(starts[okr].tolist()))
    return [{"source": "MR1", "start_sec": s1}, {"source": "MR2", "start_sec": s2},
            {"source": "R", "start_sec": sr}], None


def main():
    args = parse_args()
    with open(args.videos, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and "use" in rows[0]:
        rows = [r for r in rows if r["use"].strip() == "1"]
    elif rows and "maybe_not_play" in rows[0]:
        rows = [r for r in rows if r["maybe_not_play"].strip() != "1"]
    if args.splits:
        rows = [r for r in rows if r.get("split") in args.splits]
    if args.genres:
        rows = [r for r in rows if r.get("genre") in args.genres]
    rng = random.Random(args.seed)
    rng.shuffle(rows)  # --n や --max_per_channel で切るときに、先頭の動画ばかり選ばれないように

    questions, skipped, per_channel = [], Counter(), Counter()
    with h5py.File(f"{args.mosu_dir}/mosu_gt.h5", "r") as gt:
        for r in rows:
            if args.n and len(questions) >= args.n:
                break
            ch = r.get("channel", "")
            if args.max_per_channel and ch and per_channel[ch] >= args.max_per_channel:
                skipped["チャンネルの上限"] += 1
                continue
            if r["video_id"] not in gt:
                skipped["正解がない"] += 1
                continue
            cands, why = candidates(gt[r["video_id"]]["gt_score"][()], args.clip, args.gap, rng)
            if why:
                skipped[why] += 1
                continue
            genre = r.get("genre") or (r.get("genre_tags") or "").split("|")[0] or "不明"
            per_channel[ch] += 1
            questions.append({"question_id": f"q{len(questions) + 1:02d}", "video_id": r["video_id"],
                              "youtube_id": r["youtube_id"], "title": r.get("title", ""), "genre": genre,
                              "split": "holdout" if r.get("split") == "train" else r.get("split", ""),
                              "channel": ch, "candidates": cands})

    out = args.out or f"questions_{args.set_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"set_id": args.set_id, "clip_sec": args.clip, "gap_sec": args.gap, "questions": questions},
                  f, ensure_ascii=False, indent=1)
    print(f"問題 {len(questions)}問 → {out}（元の候補 {len(rows)}本）")
    if skipped:
        print("使わなかった動画:", dict(skipped))
    print("ジャンル:", dict(Counter(q["genre"] for q in questions).most_common()))
    print("チャンネル（多い順に5つ）:", dict(Counter(q["channel"] for q in questions).most_common(5)))


if __name__ == "__main__":
    main()
