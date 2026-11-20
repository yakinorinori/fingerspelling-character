"""
手順3・4：動作状態モデルと統一モデル（Transformer）の学習と評価

  --task dynamic : 動作指文字のすべての文字（--sokuon で促音も1クラスとして入れる）
                   特徴量は H3+O+M（本命）と H1+M（既存に近い形）を比較
  --task unified : 静止指文字＋動作指文字（＋促音）を1つのモデルで認識
                   特徴量は S+O+M（本命）と S+H3+O+M（手型を直接入れたとき）を比較
                   S は train_static_svm.py が保存した「その動画を学習していない SVM」の確率

分割は train_static_svm.py が決めた cache/folds.csv を使う（全モデルで同じ分割）。
fold k をテスト、fold k+1 を最良エポックを選ぶ検証用、残りを学習に使う。

出力（results/）:
  <task>_summary.csv, <task>_confusion_<model>_<特徴量>.png
  models/<task>_<model>_<特徴量>.pt      --final を付けたとき（recognize_video.py で使う）

使い方:
    python train_sequence.py --task dynamic
    python train_sequence.py --task unified --final
    python train_sequence.py --task dynamic --sokuon       # 促音も入れる（check_depth.py で確認してから）
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac(MPS)で未対応の処理は CPU で計算
import csv
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from common import CACHE_DIR, MODEL_DIR, RESULT_DIR, SOKUON_LABEL
from dataset import load_all, read_folds
from features import build, parse_set
from models import MODELS, build_model, count_params
from plotting import plot_confusion

DEFAULT_FEATURES = {"dynamic": ["H3+O+M", "H1+M"], "unified": ["S+O+M", "S+H3+O+M"]}


class Scaler:
    def fit(self, seqs, valids):
        x = np.concatenate([s[v] for s, v in zip(seqs, valids)])
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self

    def __call__(self, seq):
        return ((seq - self.mean) / self.std).astype(np.float32)


def batches(seqs, labels, idx, batch_size, shuffle):
    idx = np.array(idx)
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        T = max(len(seqs[j]) for j in b)
        x = np.zeros((len(b), T, seqs[b[0]].shape[1]), dtype=np.float32)
        mask = np.zeros((len(b), T), dtype=bool)
        for k, j in enumerate(b):
            x[k, :len(seqs[j])] = seqs[j]
            mask[k, :len(seqs[j])] = True
        yield torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(labels[b])


def predict(model, seqs, idx, batch_size, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, m, _ in batches(seqs, np.zeros(len(seqs), dtype=np.int64), idx, batch_size, False):
            out.append(torch.softmax(model(x.to(device), m.to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def train_one(model_name, raw, valids, labels, n_classes, tr, va, args, device):
    scaler = Scaler().fit([raw[i] for i in tr], [valids[i] for i in tr])
    seqs = [scaler(s) for s in raw]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(model_name, seqs[0].shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = np.bincount(labels[tr], minlength=n_classes).astype(np.float32)
    weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    best = (-1.0, 0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, m, y in batches(seqs, labels, tr, args.batch, True):
            opt.zero_grad()
            loss_fn(model(x.to(device), m.to(device)), y.to(device)).backward()
            opt.step()
        acc = accuracy_score(labels[va], predict(model, seqs, va, args.batch, device).argmax(1)) if len(va) else epoch
        if acc > best[0]:
            best = (acc, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model, scaler, seqs, best[1]


def load_oof(rows, name):
    path = CACHE_DIR / f"svm_oof_{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} がありません。先に train_static_svm.py を実行してください")
    d = np.load(path)
    table = {v: d[f"p{i}"] for i, v in enumerate(d["videos"])}
    return [table[r["video"]] for r in rows], list(d["classes"])


def main():
    parser = argparse.ArgumentParser(description="動作状態モデル・統一モデルの学習と評価")
    parser.add_argument("--task", required=True, choices=["dynamic", "unified"])
    parser.add_argument("--features", nargs="+", help="省略時: dynamic は H3+O+M, H1+M / unified は S+O+M, S+H3+O+M")
    parser.add_argument("--models", nargs="+", default=["transformer"], choices=list(MODELS))
    parser.add_argument("--sokuon", action="store_true", help="促音を1クラスとして入れる")
    parser.add_argument("--svm-features", default="H3+O", help="S に使う SVM（train_static_svm.py の本命）")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--final", action="store_true", help="比較のあと、全データで学習したモデルを保存する")
    args = parser.parse_args()
    feature_sets = args.features or DEFAULT_FEATURES[args.task]

    device = torch.device(("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                           else "cpu") if args.device == "auto" else args.device)
    kinds = ["dynamic"] if args.task == "dynamic" else ["static", "dynamic"]
    if args.sokuon:
        kinds.append("sokuon")
    rows, blocks, valids = load_all(kinds)
    fold, n_folds = read_folds(rows)
    classes = sorted({r["label"] for r in rows}, key=lambda c: (c == SOKUON_LABEL, c))
    labels = np.array([classes.index(r["label"]) for r in rows])
    kind = np.array([r["kind"] for r in rows])
    print(f"{args.task}: 動画 {len(rows)} 本, {len(classes)} クラス（{''.join(c for c in classes if len(c) == 1)}"
          f"{' ＋促音' if SOKUON_LABEL in classes else ''}）, {n_folds} 分割, デバイス {device}")

    extra = None
    if any("S" in parse_set(fs) for fs in feature_sets):
        oof, svm_classes = load_oof(rows, args.svm_features)
        extra = [{"S": p} for p in oof]

    RESULT_DIR.mkdir(exist_ok=True)
    summary = []
    for fs in feature_sets:
        raw = [build(fs, b, extra[i] if extra else None) for i, b in enumerate(blocks)]
        for model_name in args.models:
            name = f"{args.task}_{model_name}_{fs}"
            t0 = time.time()
            true, pred, kinds_test, accs = [], [], [], []
            for k in range(n_folds):
                te = np.flatnonzero(fold == k)
                va = np.flatnonzero(fold == (k + 1) % n_folds)
                tr = np.flatnonzero((fold != k) & (fold != (k + 1) % n_folds))
                model, _, seqs, ep = train_one(model_name, raw, valids, labels, len(classes), tr, va, args, device)
                p = predict(model, seqs, te, args.batch, device).argmax(1)
                true += list(labels[te])
                pred += list(p)
                kinds_test += list(kind[te])
                accs.append(accuracy_score(labels[te], p))
                print(f"{name:32s} fold {k + 1}/{n_folds}: 正解率 {accs[-1]:.3f}（最良エポック {ep}）")
            true, pred, kinds_test = np.array(true), np.array(pred), np.array(kinds_test)
            s = {"task": args.task, "model": model_name, "features": fs, "dim": raw[0].shape[1],
                 "accuracy": accuracy_score(true, pred), "accuracy_std": float(np.std(accs)),
                 "macro_f1": f1_score(true, pred, average="macro"),
                 "params": count_params(build_model(model_name, raw[0].shape[1], len(classes))),
                 "train_sec": round(time.time() - t0, 1)}
            for kd in ["static", "dynamic", "sokuon"]:     # 統一モデルでは静止・動作それぞれの正解率も出す
                if (kinds_test == kd).any() and len(set(kinds_test)) > 1:
                    s[f"accuracy_{kd}"] = accuracy_score(true[kinds_test == kd], pred[kinds_test == kd])
            summary.append(s)
            plot_confusion(true, pred, classes, f"{name}  acc {s['accuracy']:.3f}",
                           RESULT_DIR / f"{args.task}_confusion_{model_name}_{fs}.png")

            if args.final:
                va = np.flatnonzero(fold == 0)
                tr = np.flatnonzero(fold != 0)
                model, scaler, _, ep = train_one(model_name, raw, valids, labels, len(classes), tr, va, args, device)
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"task": args.task, "model": model_name, "features": fs, "classes": classes,
                            "in_dim": int(raw[0].shape[1]), "state_dict": model.state_dict(),
                            "mean": scaler.mean, "std": scaler.std, "svm_features": args.svm_features,
                            "cv_accuracy": s["accuracy"]}, MODEL_DIR / f"{name}.pt")
                print(f"モデルを保存: results/models/{name}.pt")

    keys = sorted({k for s in summary for k in s}, key=lambda k: list(summary[0]).index(k) if k in summary[0] else 99)
    with open(RESULT_DIR / f"{args.task}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"\n=== {args.task} の結果 ===")
    for s in summary:
        extra_acc = "".join(f", {k[9:]} {s[k]:.3f}" for k in s if k.startswith("accuracy_") and k != "accuracy_std")
        print(f"{s['model']:12s} {s['features']:10s} {s['dim']:3d}次元  正解率 {s['accuracy']:.3f} ± {s['accuracy_std']:.3f}"
              f", マクロF1 {s['macro_f1']:.3f}{extra_acc}  （{s['params']:,} パラメータ, {s['train_sec']} 秒）")


if __name__ == "__main__":
    main()
