"""
手順2：静止状態モデル（SVM）の学習と評価

  - 静止指文字の動画から「手が見えているフレームの中央の半分」をすべて使う（手を出す途中・下ろす途中を除く）
    データが多くて時間がかかりすぎるときは --frames-per-video 20 などで1本あたりの枚数を減らす
  - 特徴量は H3+O（本命）と H1（既存・基準）を比較する
  - 全動画（静止・動作・促音）に fold 番号を振って cache/folds.csv に保存する。以降のモデルも同じ分割を使う
  - 統一モデル用に、各動画の全フレームの SVM の確率を「その動画を学習していない SVM」で計算して保存する
    → cache/svm_oof_<特徴量>.npz

出力（results/）:
  svm_summary.csv, svm_confusion_<特徴量>.png
  models/svm_<特徴量>.joblib     全静止データで学習した SVM（recognize_video.py で使う）

使い方:
    python train_static_svm.py
    python train_static_svm.py --group-regex "^(ASL_[A-E]|ubmoji)"   # ファイル名から人を判別して人ごとに分割
"""
import argparse
import csv
import warnings

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CACHE_DIR, MODEL_DIR, RESULT_DIR
from dataset import load_all, save_folds
from features import build
from plotting import plot_confusion

# scikit-learn 1.9 以降で出る SVC(probability=True) の予告警告を表示しない（1.10 までは動作に影響なし）
warnings.filterwarnings("ignore", message=".*probability.*deprecated.*", category=FutureWarning)


def center_frames(valid, ratio=0.5):
    """手が見えているフレームのうち、中央の ratio の範囲"""
    idx = np.flatnonzero(valid)
    n = len(idx)
    cut = int(round(n * (1 - ratio) / 2))
    return idx[cut:n - cut] if n - 2 * cut > 0 else idx


def pick(idx, k):
    if k <= 0 or len(idx) <= k:
        return idx
    return idx[np.linspace(0, len(idx) - 1, k).round().astype(int)]


def make_svm(C):
    return make_pipeline(StandardScaler(), SVC(C=C, gamma="scale", probability=True, cache_size=2000,
                                               class_weight="balanced", random_state=0))


def training_frames(ids, feats, valids, labels, k):
    X, y = [], []
    for i in ids:
        f = pick(center_frames(valids[i]), k)
        X.append(feats[i][f])
        y += [labels[i]] * len(f)
    return np.concatenate(X), np.array(y)


def full_probs(svm, feat, valid, n_classes):
    """全フレームの確率（手が無いフレームは 0）"""
    out = np.zeros((len(feat), n_classes), dtype=np.float32)
    if valid.any():
        out[valid] = svm.predict_proba(feat[valid])
    return out


def main():
    parser = argparse.ArgumentParser(description="静止状態モデル（SVM）の学習と評価")
    parser.add_argument("--features", nargs="+", default=["H3+O", "H1"], help="最初のものが本命（統一モデルに使う）")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-regex", default="", help="ファイル名から人などを取り出す正規表現。省略時は動画ごと")
    parser.add_argument("--frames-per-video", type=int, default=0,
                        help="1本の動画から学習に使うフレーム数。0（既定）で中央の半分のフレームをすべて使う")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="特徴量を計算し直す")
    args = parser.parse_args()

    rows, blocks, valids = load_all(refresh=args.refresh)
    fold, n_folds = save_folds(rows, args.folds, args.group_regex, args.seed)
    static = np.array([i for i, r in enumerate(rows) if r["kind"] == "static"])
    classes = sorted({rows[i]["label"] for i in static})
    cls_index = {c: k for k, c in enumerate(classes)}
    labels = np.array([cls_index.get(r["label"], -1) for r in rows])
    print(f"動画 {len(rows)} 本（静止 {len(static)} 本, {len(classes)} 文字）, {n_folds} 分割")

    RESULT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for fs_i, fs in enumerate(args.features):
        feats = [build(fs, b) for b in blocks]
        oof = [None] * len(rows) if fs_i == 0 else None
        vid_true, vid_pred, fr_true, fr_pred, accs = [], [], [], [], []
        for k in range(n_folds):
            tr = static[fold[static] != k]
            te = static[fold[static] == k]
            X, y = training_frames(tr, feats, valids, labels, args.frames_per_video)
            if k == 0:
                print(f"SVM {fs:8s} 学習フレーム数 {len(X)}（1 fold あたり）")
            svm = make_svm(args.C).fit(X, y)
            for i in te:                       # 動画単位：中央の半分のフレームの確率の平均
                c = center_frames(valids[i])
                p = svm.predict_proba(feats[i][c])
                fr_true += [labels[i]] * len(c)
                fr_pred += list(svm.classes_[p.argmax(1)])
                vid_true.append(labels[i])
                vid_pred.append(svm.classes_[p.mean(0).argmax()])
            accs.append(accuracy_score([labels[i] for i in te], vid_pred[-len(te):]) if len(te) else np.nan)
            if oof is not None:                # 統一モデル用：この fold の全動画（静止以外も）の確率
                for i in np.flatnonzero(fold == k):
                    p = np.zeros((len(feats[i]), len(classes)), dtype=np.float32)
                    p[:, svm.classes_] = full_probs(svm, feats[i], valids[i], len(svm.classes_))
                    oof[i] = p
            print(f"SVM {fs:8s} fold {k + 1}/{n_folds}: 動画単位の正解率 {accs[-1]:.3f}")

        s = {"features": fs, "dim": feats[0].shape[1],
             "video_accuracy": accuracy_score(vid_true, vid_pred),
             "video_macro_f1": f1_score(vid_true, vid_pred, average="macro"),
             "frame_accuracy": accuracy_score(fr_true, fr_pred),
             "fold_accuracy_std": float(np.nanstd(accs))}
        summary.append(s)
        plot_confusion(vid_true, vid_pred, classes, f"SVM {fs}  動画単位 acc {s['video_accuracy']:.3f}",
                       RESULT_DIR / f"svm_confusion_{fs}.png")

        if oof is not None:
            np.savez_compressed(CACHE_DIR / f"svm_oof_{fs}.npz", classes=np.array(classes),
                                videos=np.array([r["video"] for r in rows]),
                                **{f"p{i}": p for i, p in enumerate(oof)})
            print(f"統一モデル用の確率を保存: cache/svm_oof_{fs}.npz")

        X, y = training_frames(static, feats, valids, labels, args.frames_per_video)
        final = make_svm(args.C).fit(X, y)
        joblib.dump({"svm": final, "features": fs, "classes": classes}, MODEL_DIR / f"svm_{fs}.joblib")
        print(f"全静止データで学習した SVM を保存: results/models/svm_{fs}.joblib\n")

    with open(RESULT_DIR / "svm_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("=== 静止SVM の結果 ===")
    for s in summary:
        print(f"{s['features']:8s} {s['dim']:3d}次元  動画単位 正解率 {s['video_accuracy']:.3f}"
              f"（fold間の標準偏差 {s['fold_accuracy_std']:.3f}）, マクロF1 {s['video_macro_f1']:.3f}, "
              f"フレーム単位 正解率 {s['frame_accuracy']:.3f}")


if __name__ == "__main__":
    main()
