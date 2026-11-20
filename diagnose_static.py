"""
静止SVM の精度が伸びない原因を調べる（train_static_svm.py の後に実行。分割は cache/folds.csv をそのまま使う）

調べること:
  1. データの出どころ（ASL_A〜E / ubmoji / 自前）ごとの正解率   → 特定のデータセットだけ低くないか
  2. 間違えやすい文字の組（上位）                                → 形が似ている文字か、ラベルの付け方の問題か
  3. 特徴量ブロックごとの正解率（H3 / O / H3+O）                 → 向き O が足を引っ張っていないか（左右の手の混在など）
  4. C を変えたときの正解率                                    → 設定の問題か

出力: results/diagnose_static.csv（動画ごとの正解・予測・出どころ）と画面表示

使い方:
    python diagnose_static.py
    python diagnose_static.py --source-regex "^(ASL_[A-E]|ubmoji)"   # 出どころの見分け方（ファイル名の正規表現）
    python diagnose_static.py --skip-grid                            # 4. を飛ばす（時間がかかるため）
"""
import argparse
import csv
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from common import RESULT_DIR
from dataset import load_all, read_folds
from features import build
from train_static_svm import center_frames, make_svm, training_frames

warnings.filterwarnings("ignore", message=".*probability.*deprecated.*", category=FutureWarning)


def source_of(video, pat):
    m = pat.search(Path(video).name)
    if not m:
        return "自前"
    return m.group(1) if m.groups() else m.group(0)


def evaluate(fs, rows, blocks, valids, labels, static, fold, n_folds, C, k_frames):
    feats = [build(fs, b) for b in blocks]
    pred = {}
    for k in range(n_folds):
        tr = static[fold[static] != k]
        te = static[fold[static] == k]
        if len(te) == 0:
            continue
        X, y = training_frames(tr, feats, valids, labels, k_frames)
        svm = make_svm(C).fit(X, y)
        for i in te:
            p = svm.predict_proba(feats[i][center_frames(valids[i])])
            pred[i] = svm.classes_[p.mean(0).argmax()]
    idx = sorted(pred)
    return idx, np.array([pred[i] for i in idx])


def main():
    parser = argparse.ArgumentParser(description="静止SVM の精度が伸びない原因を調べる")
    parser.add_argument("--source-regex", default=r"^(ASL_[A-E]|ubmoji)")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--frames-per-video", type=int, default=0)
    parser.add_argument("--top", type=int, default=15, help="表示する「間違えやすい組」の数")
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()

    rows, blocks, valids = load_all()
    fold, n_folds = read_folds(rows)
    static = np.array([i for i, r in enumerate(rows) if r["kind"] == "static"])
    classes = sorted({rows[i]["label"] for i in static})
    cls_index = {c: k for k, c in enumerate(classes)}
    labels = np.array([cls_index.get(r["label"], -1) for r in rows])
    pat = re.compile(args.source_regex)
    source = np.array([source_of(r["video"], pat) for r in rows])

    print("=== データの内訳（静止）===")
    for s, n in sorted(Counter(source[static]).items()):
        n_cls = len({labels[i] for i in static if source[i] == s})
        print(f"  {s:8s} {n:5d} 本, {n_cls} 文字")
    counts = Counter(rows[i]["label"] for i in static)
    few = sorted(counts.items(), key=lambda x: x[1])[:5]
    print("  本数が少ない文字: " + ", ".join(f"{c}={n}" for c, n in few))

    # --- 1, 2: 本命 H3+O の結果を詳しく見る ---
    idx, pred = evaluate("H3+O", rows, blocks, valids, labels, static, fold, n_folds,
                         args.C, args.frames_per_video)
    true = labels[idx]
    src = source[idx]
    print(f"\n=== 1. 出どころごとの正解率（H3+O）===  全体 {accuracy_score(true, pred):.3f}")
    for s in sorted(set(src)):
        m = src == s
        print(f"  {s:8s} {accuracy_score(true[m], pred[m]):.3f}  ({m.sum()} 本)")

    print(f"\n=== 2. 間違えやすい組（正解 → 予測, 本数）上位 {args.top} ===")
    wrong = Counter((classes[t], classes[p]) for t, p in zip(true, pred) if t != p)
    for (t, p), n in wrong.most_common(args.top):
        by_src = Counter(src[(true == cls_index[t]) & (pred == cls_index[p])])
        detail = ", ".join(f"{s}:{c}" for s, c in by_src.most_common())
        print(f"  {t} → {p}: {n} 本  [{detail}]")
    per_cls = {c: accuracy_score(true[true == k], pred[true == k]) for k, c in enumerate(classes) if (true == k).any()}
    worst = sorted(per_cls.items(), key=lambda x: x[1])[:8]
    print("  正解率が低い文字: " + ", ".join(f"{c}={a:.2f}" for c, a in worst))

    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / "diagnose_static.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "source", "true", "pred", "correct"])
        for i, p in zip(idx, pred):
            w.writerow([rows[i]["video"], source[i], classes[labels[i]], classes[p], int(labels[i] == p)])
    print("動画ごとの結果: results/diagnose_static.csv")

    # --- 3: ブロックごと ---
    print("\n=== 3. 特徴量ブロックごとの正解率（出どころ別）===")
    for fs in ["H3", "O", "H3+O"]:
        i2, p2 = evaluate(fs, rows, blocks, valids, labels, static, fold, n_folds, args.C, args.frames_per_video)
        t2, s2 = labels[i2], source[i2]
        parts = "  ".join(f"{s}:{accuracy_score(t2[s2 == s], p2[s2 == s]):.3f}" for s in sorted(set(s2)))
        print(f"  {fs:5s} 全体 {accuracy_score(t2, p2):.3f}   {parts}")

    # --- 4: 設定 ---
    if not args.skip_grid:
        print("\n=== 4. C を変えたとき（H3+O, 1本あたり20フレームで速く比べる, 全体の正解率）===")
        for k_frames in [20]:
            accs = []
            for C in [1, 10, 100]:
                i2, p2 = evaluate("H3+O", rows, blocks, valids, labels, static, fold, n_folds, C, k_frames)
                accs.append(f"C={C:<3d} {accuracy_score(labels[i2], p2):.3f}")
            print(f"  フレーム {k_frames:2d} 枚/本:  " + "   ".join(accs))


if __name__ == "__main__":
    main()
