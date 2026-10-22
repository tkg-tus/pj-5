"""
train.py ― 見どころ予測の学習スクリプト（共通の雛形）

使い方: python train.py --system fast --modality visual --seed 0
入力:   data/features/{modality}/{video_id}.npy  形（秒数, 次元）
        data/labels/{video_id}.npy               形（秒数,） 1秒ごとの正解（0〜1）
        data/splits/{train,val}.txt              1行に1つの動画ID
出力:   runs/{system}_{modality}_seed{n}/
          best.pt       valの誤差が一番小さかった時点のモデル
          pred_val.csv  予測（video_id, t, score）scoreは動画ごとに0〜1
          summary.json  エポックごとの記録、最良の成績、学習時間、パラメータ数、GPU

★各自で実装: systems/{system}.py に build_model と train_one_epoch（それ以外は共通）
"""
import argparse, importlib, json, os, random, time
import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)  # 自分のシステム名
    p.add_argument("--modality", required=True, choices=["visual", "audio", "text"])
    p.add_argument("--seed", type=int, required=True)  # 0, 1, 2
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--lr", type=float, default=1e-3)  # 全員共通の固定値
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--subset", type=int, default=0)  # 動作確認用に動画数を絞る（0なら全部）
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(split, args):
    """[(動画ID, 特徴量, 正解), ...] を返す。秒数は短い方に合わせる"""
    with open(f"{args.data_dir}/splits/{split}.txt") as f:
        ids = [l.strip() for l in f if l.strip()]
    if args.subset:
        ids = ids[:args.subset]
    data = []
    for vid in ids:
        x = np.load(f"{args.data_dir}/features/{args.modality}/{vid}.npy").astype(np.float32)
        y = np.load(f"{args.data_dir}/labels/{vid}.npy").astype(np.float32)
        T = min(len(x), len(y))
        data.append((vid, x[:T], y[:T]))
    return data


def batches(data, batch_size, dev, rng=None):
    """バッチごとに (元データ, X, Y, M) を返す。rngを渡すとシャッフルする
    X:（バッチ, 秒数, 次元） Y:（バッチ, 秒数） M: 有効な秒がTrue（詰め物はFalse）"""
    idx = rng.permutation(len(data)) if rng is not None else np.arange(len(data))
    for i in range(0, len(idx), batch_size):
        b = [data[j] for j in idx[i:i + batch_size]]
        T = max(len(x) for _, x, _ in b)
        X = torch.zeros(len(b), T, b[0][1].shape[1]); Y = torch.zeros(len(b), T)
        M = torch.zeros(len(b), T, dtype=torch.bool)
        for k, (_, x, y) in enumerate(b):
            X[k, :len(x)] = torch.from_numpy(x); Y[k, :len(y)] = torch.from_numpy(y); M[k, :len(x)] = True
        yield b, X.to(dev), Y.to(dev), M.to(dev)


def load_system(name):
    """systems/{name}.py を読み込む。build_model と train_one_epoch を各自がそこに実装する"""
    return importlib.import_module(f"systems.{name}")


@torch.no_grad()
def predict(model, data, args, dev):
    model.eval()
    out = {}
    for b, X, _, M in batches(data, args.batch_size, dev):
        P = model(X, M).float().cpu().numpy()
        for k, (vid, x, _) in enumerate(b):
            out[vid] = P[k, :len(x)]
    return out


def val_error(preds, data):
    """学習を止める判断用の値（小さいほど良い）。評価方法が決まったら、ここだけ差し替える。
    仮の方法：予測を動画ごとに0〜1へ正規化し、正解との二乗誤差を動画ごとに出して平均する。
    （正規化するのは、各自の損失の違いで出力の大きさが変わっても、同じ基準で比べるため）"""
    errs = []
    for vid, _, y in data:
        p = preds[vid]
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        errs.append(float(np.mean((p - y) ** 2)))
    return float(np.mean(errs))


def save_pred_csv(preds, path):
    with open(path, "w") as f:
        f.write("video_id,t,score\n")
        for vid, p in preds.items():
            s = (p - p.min()) / (p.max() - p.min() + 1e-8)  # 動画ごとに0〜1
            f.writelines(f"{vid},{t},{v:.6f}\n" for t, v in enumerate(s))


def main():
    args = parse_args()
    system = load_system(args.system)
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU接続（本番は共通のuv環境で）
    out = f"{args.out_dir}/{args.system}_{args.modality}_seed{args.seed}"
    os.makedirs(out, exist_ok=True)

    train, val = load_data("train", args), load_data("val", args)
    model = system.build_model(train[0][1].shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    best, best_epoch, wait, log, t_start = float("inf"), -1, 0, [], time.time()
    for epoch in range(args.max_epochs):
        t0 = time.time()
        loss = system.train_one_epoch(model, train, opt, args, dev, rng)
        err = val_error(predict(model, val, args, dev), val)
        log.append({"epoch": epoch, "loss": loss, "val_error": err, "sec": round(time.time() - t0, 1)})
        print(log[-1])
        if err < best:  # valの誤差が下がったら保存
            best, best_epoch, wait = err, epoch, 0
            torch.save(model.state_dict(), f"{out}/best.pt")
        else:
            wait += 1
            if wait >= args.patience:  # 5回改善されなければ終了
                break

    model.load_state_dict(torch.load(f"{out}/best.pt"))
    save_pred_csv(predict(model, val, args, dev), f"{out}/pred_val.csv")
    with open(f"{out}/summary.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_error": best,
                   "train_sec": round(time.time() - t_start, 1),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                   "args": vars(args), "log": log}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
