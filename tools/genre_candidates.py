"""
genre_candidates.py ― 人手評価に使うジャンル（タイトル）の候補を、本数・長さ・YouTube で見られるかで並べる

使い方: python tools/genre_candidates.py                        （ゲームとスポーツのタグを、本数の多い順に）
        python tools/genre_candidates.py --tags "League of Legends" "Call of Duty" --check 40
入力:   mosu/mosu_metadata.csv, mosu/mosu_split.json, mosu/vocabulary.csv
出力:   画面に表。--check N を付けると、各タグの val と test から N 本ずつ選び、
        YouTube でまだ見られるか（公開のままか）を確かめて、その割合を出す（ネットが必要）
"""
import argparse, ast, csv, json, urllib.error, urllib.request
from collections import Counter, defaultdict
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mosu_dir", default="mosu")
    p.add_argument("--tags", nargs="*", default=None)  # 調べるタグ。指定なしならゲームとスポーツの上位
    p.add_argument("--verticals", nargs="*", default=["Games", "Sports"])
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--min_val", type=int, default=20)  # val がこれより少ないタグは出さない
    p.add_argument("--check", type=int, default=0)  # YouTube で見られるかを確かめる本数（val・test それぞれ）
    return p.parse_args()


def watchable(youtube_id):
    """oEmbed（埋め込み用の公開の窓口）で、動画が公開のままかを確かめる。削除・非公開なら 4xx が返る"""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={youtube_id}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=10):
            return True
    except urllib.error.HTTPError:
        return False


def main():
    args = parse_args()
    with open(f"{args.mosu_dir}/vocabulary.csv", encoding="utf-8") as f:
        voc = {int(r["Index"]): r for r in csv.DictReader(f)}
    with open(f"{args.mosu_dir}/mosu_metadata.csv", encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))
    split = json.load(open(f"{args.mosu_dir}/mosu_split.json"))
    split_of = {v: n for n in ("train", "val", "test") for v in split[f"{n}_keys"]}

    counts, durs, ids = defaultdict(Counter), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in meta:
        s = split_of.get(r["video_id"])
        for t in ast.literal_eval(r["labels"]):
            counts[t][s] += 1
            durs[t].append(int(r["duration"]))
            ids[t][s].append(r["youtube_id"])

    if args.tags:
        index_of = {r["Name"]: i for i, r in voc.items()}
        missing = [t for t in args.tags if t not in index_of]
        if missing:
            raise SystemExit(f"見つからないタグ: {missing}")
        chosen = [index_of[t] for t in args.tags]
    else:
        generic = {"Game", "Video game", "Sports", "Gameplay", "Games"}
        chosen = [t for t in counts if voc.get(t, {}).get("Vertical1") in args.verticals
                  and voc[t]["Name"] not in generic and counts[t]["val"] >= args.min_val]
        chosen = sorted(chosen, key=lambda t: -sum(counts[t].values()))[:args.top]

    rng = np.random.default_rng(0)
    print(f"{'タグ':34s} {'分類':8s} {'train':>6s} {'val':>5s} {'test':>5s} {'長さ中央値':>8s} {'10分以下':>7s}"
          + ("  見られる割合(val/test)" if args.check else ""))
    for t in chosen:
        c, d = counts[t], np.array(durs[t])
        line = (f"{voc[t]['Name'][:34]:34s} {voc[t].get('Vertical1', '')[:8]:8s} {c['train']:6d} {c['val']:5d} {c['test']:5d}"
                f" {int(np.median(d)):7d}秒 {np.mean(d <= 600):7.0%}")
        if args.check:
            rates = []
            for s in ("val", "test"):
                pool = ids[t][s]
                sample = rng.choice(pool, size=min(args.check, len(pool)), replace=False) if pool else []
                rates.append(np.mean([watchable(y) for y in sample]) if len(sample) else float("nan"))
            line += f"  {rates[0]:.0%} / {rates[1]:.0%}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
