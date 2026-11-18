"""
find_ja_videos.py ― ゲーム（cluster_id 1）の動画から、タイトルが日本語のものを探して、カテゴリごとに集計する

使い方: python tools/find_ja_videos.py                 （タイトルを取ってきて、集計まで）
        python tools/find_ja_videos.py --analyze_only  （取ってきた結果だけで集計し直す）
  時間がかかる（1万本強で30分〜1時間）ので、②のログインノードで tmux の中で動かす。
  途中で止まっても、もう一度同じコマンドを実行すれば続きから取る。

入力:   mosu/mosu_metadata.csv, mosu/mosu_split.json, mosu/vocabulary.csv
出力:   mosu/oembed_cache.csv     取ってきたタイトルとチャンネル名（全部の動画。見られない動画は ok=0）
        ja_game_videos.csv        日本語の動画の一覧（split・タグ・ジャンル・プレイ動画でなさそうかの印つき）
        画面に、split ごとの本数、ゲームのジャンル別・タイトル別・チャンネル別の本数
"""
import argparse, ast, csv, json, os, re, time, urllib.error, urllib.request
from collections import Counter

KANA = re.compile(r"[぀-ヿ]")  # ひらがな・カタカナ（漢字だけだと中国語も入るので、かなで判定）
# プレイ動画でなさそうなタイトル（PV、紹介映像、インタビューなど）。印を付けるだけで、外しはしない
NOT_PLAY = re.compile(r"PV|CM|トレーラー|Trailer|プロモーション|紹介映像|インタビュー|Developer|"
                      r"開発者|発表|ティザー|Teaser|オープニング|Opening|OP\b|ED\b|MAD|AMV", re.I)
GENERIC = {"Game", "Video game", "Games", "Gameplay", "PC game", "Mobile game"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mosu_dir", default="mosu")
    p.add_argument("--cluster", default="1")
    p.add_argument("--out", default="ja_game_videos.csv")
    p.add_argument("--sleep", type=float, default=0.2)  # 問い合わせの間隔（短すぎると YouTube に止められる）
    p.add_argument("--analyze_only", action="store_true")
    p.add_argument("--top", type=int, default=20)
    return p.parse_args()


def oembed(youtube_id):
    """(見られるか, タイトル, チャンネル名)。消された・非公開・埋め込み禁止の動画は 4xx が返る"""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={youtube_id}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.load(r)
        return 1, d.get("title", ""), d.get("author_name", "")
    except urllib.error.HTTPError:
        return 0, "", ""


def fetch(videos, cache_path, sleep):
    """まだ取っていない動画だけ取る。1本ごとにファイルに書くので、止まっても続きから再開できる"""
    done = set()
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            done = {r["video_id"] for r in csv.DictReader(f)}
    todo = [v for v in videos if v["video_id"] not in done]
    print(f"タイトル取得: 全{len(videos)}本のうち、取得済み {len(done)}本、残り {len(todo)}本", flush=True)
    new = not os.path.exists(cache_path)
    with open(cache_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["video_id", "youtube_id", "ok", "title", "channel"])
        for i, v in enumerate(todo):
            for attempt in range(3):  # ネットの一時的な失敗は少し待ってやり直す
                try:
                    ok, title, channel = oembed(v["youtube_id"])
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                print(f"取得できず（あとで再実行すれば取り直す）: {v['video_id']}", flush=True)
                continue
            w.writerow([v["video_id"], v["youtube_id"], ok, title, channel])
            f.flush()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
            time.sleep(sleep)


def table(counter, top, total, head):
    print(f"\n===== {head} =====")
    for name, n in counter.most_common(top):
        print(f"{n:5d}本 {n / total:6.1%}  {name}")


def main():
    args = parse_args()
    with open(f"{args.mosu_dir}/vocabulary.csv", encoding="utf-8") as f:
        vocab = {int(r["Index"]): r["Name"] for r in csv.DictReader(f)}
    with open(f"{args.mosu_dir}/mosu_metadata.csv", encoding="utf-8-sig") as f:
        meta = [r for r in csv.DictReader(f) if r["cluster_id"] == args.cluster]
    split = json.load(open(f"{args.mosu_dir}/mosu_split.json"))
    split_of = {v: s for s in ("train", "val", "test") for v in split[f"{s}_keys"]}
    cache_path = f"{args.mosu_dir}/oembed_cache.csv"

    if not args.analyze_only:
        fetch(meta, cache_path, args.sleep)
    with open(cache_path, encoding="utf-8") as f:
        info = {r["video_id"]: r for r in csv.DictReader(f)}

    rows = []
    for v in meta:
        c = info.get(v["video_id"])
        if not c or c["ok"] != "1" or not KANA.search(c["title"]):
            continue
        tags = [vocab.get(t, str(t)) for t in ast.literal_eval(v["labels"])]
        tags = [t for t in tags if t not in GENERIC]
        # ジャンルらしいタグ（「〜game」「shooter」で終わるもの）と、それ以外（タイトル名など）に分ける
        genres = [t for t in tags if re.search(r"(game|shooter)$", t, re.I)]
        rows.append({"split": split_of.get(v["video_id"], "?"), "video_id": v["video_id"],
                     "youtube_id": v["youtube_id"], "duration": v["duration"], "title": c["title"],
                     "channel": c["channel"], "genre_tags": "|".join(genres),
                     "other_tags": "|".join(t for t in tags if t not in genres),
                     "maybe_not_play": int(bool(NOT_PLAY.search(c["title"])))})
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:  # BOM つき：Excel で文字化けしない
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["split"])
        w.writeheader()
        w.writerows(rows)

    checked = Counter(split_of.get(v["video_id"], "?") for v in meta if v["video_id"] in info)
    seen = Counter(split_of.get(v["video_id"], "?") for v in meta if info.get(v["video_id"], {}).get("ok") == "1")
    ja = Counter(r["split"] for r in rows)
    ja_play = Counter(r["split"] for r in rows if not r["maybe_not_play"])
    print("\n===== split ごとの本数 =====")
    print(f"{'split':6s} {'調べた':>7s} {'見られる':>7s} {'日本語':>6s} {'うちプレイ動画らしい':>10s}")
    for s in ("train", "val", "test"):
        print(f"{s:6s} {checked[s]:7d} {seen[s]:7d} {ja[s]:6d} {ja_play[s]:10d}")
    if not rows:
        return
    play = [r for r in rows if not r["maybe_not_play"]]
    n = len(play)
    print(f"\n（以下は、プレイ動画らしい {n}本で集計。1本に複数のタグが付くので、合計は100%を超える）")
    table(Counter(g for r in play for g in r["genre_tags"].split("|") if g), args.top, n, "ゲームのジャンル（タグ）")
    table(Counter(t for r in play for t in r["other_tags"].split("|") if t), args.top, n, "そのほかのタグ（タイトル名など）")
    table(Counter(r["channel"] for r in play), args.top, n, "チャンネル（同じチャンネルに偏っていないか）")
    print(f"\n一覧 → {args.out}（maybe_not_play=1 はタイトルから見てプレイ動画でなさそうなもの。目で確認すること）")


if __name__ == "__main__":
    main()
