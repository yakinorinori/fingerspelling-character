"""骨格 → 特徴量ブロックの読み込み（計算結果は cache/blocks/ に保存して使い回す）"""
import csv

import numpy as np

from common import CACHE_DIR, assign_folds, load_index
from features import compute_blocks

FEATURE_VERSION = "2"            # 特徴量の計算を変えたら上げる（保存済みの計算結果を作り直す）
BLOCK_DIR = CACHE_DIR / "blocks"
FOLDS_CSV = CACHE_DIR / "folds.csv"
ALL_BLOCKS = ("H1", "H3", "O", "M")


def load_raw(row):
    d = np.load(CACHE_DIR / row["npz"])
    world = d["world"] if "world" in d.files else np.full_like(d["lm"], np.nan)
    return d["lm"], world, int(d["width"]), int(d["height"]), float(d["fps"])


def load_blocks(row, refresh=False):
    path = BLOCK_DIR / row["npz"]
    if not refresh and path.exists():
        d = np.load(path)
        if str(d["version"]) == FEATURE_VERSION:
            return {b: d[b] for b in ALL_BLOCKS}, d["valid"]
    lm, world, w, h, _ = load_raw(row)
    blocks = compute_blocks(lm, world, w, h, ALL_BLOCKS)
    valid = ~np.isnan(lm).any(axis=(1, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, version=FEATURE_VERSION, valid=valid, **blocks)
    return blocks, valid


def load_all(kinds=None, refresh=False):
    """戻り値: rows（手が1フレームも無い動画は除外）, blocks のリスト, valid のリスト"""
    rows, blocks, valids = [], [], []
    all_rows = load_index(kinds)
    for i, r in enumerate(all_rows, 1):
        b, v = load_blocks(r, refresh)
        if v.sum() == 0:
            print(f"[注意] 手が1フレームも検出できなかったので除外: {r['video']}")
            continue
        rows.append(r)
        blocks.append(b)
        valids.append(v)
        if i % 500 == 0:
            print(f"  特徴量を読み込み中… {i}/{len(all_rows)}")
    return rows, blocks, valids


def save_folds(rows, n_folds, group_regex, seed):
    fold, n = assign_folds(rows, n_folds, group_regex, seed)
    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FOLDS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "fold", "n_folds", "group_regex", "seed"])
        for r, k in zip(rows, fold):
            w.writerow([r["video"], k, n, group_regex, seed])
    return fold, n


def read_folds(rows):
    """train_static_svm.py が決めた分割を読む（全モデルで同じ分割を使うため）"""
    if not FOLDS_CSV.exists():
        raise SystemExit(f"{FOLDS_CSV} がありません。先に train_static_svm.py を実行してください")
    with open(FOLDS_CSV, encoding="utf-8") as f:
        table = {r["video"]: (int(r["fold"]), int(r["n_folds"])) for r in csv.DictReader(f)}
    missing = [r["video"] for r in rows if r["video"] not in table]
    if missing:
        raise SystemExit(f"分割が決まっていない動画が {len(missing)} 本あります（例: {missing[0]}）。"
                         "train_static_svm.py をもう一度実行してください")
    fold = np.array([table[r["video"]][0] for r in rows])
    return fold, table[rows[0]["video"]][1]
