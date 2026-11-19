"""単体文字認識 共通：パス、フォルダ探索、データ一覧、分割（fold）"""
import csv
import re
import unicodedata
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
# Fingerspelling-repos/fingerspelling-character/ から見た Fingerspelling/Video/transformer訓練
DEFAULT_TRAIN_DIR = SCRIPT_DIR.parent.parent / "Fingerspelling" / "Video" / "transformer訓練"
GROUP_DIRS = {"static": "静止指文字", "dynamic": "動作指文字", "sokuon": "促音"}
SOKUON_LABEL = "促音"

CACHE_DIR = SCRIPT_DIR / "cache"
LANDMARK_DIR = CACHE_DIR / "landmarks"
INDEX_CSV = CACHE_DIR / "index.csv"
RESULT_DIR = SCRIPT_DIR / "results"
MODEL_DIR = RESULT_DIR / "models"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}


def norm(name):
    """macOS の濁点の分解（NFD）と全角・半角スペースの違いを吸収する"""
    return unicodedata.normalize("NFC", name).replace("　", "").replace(" ", "")


def resolve_dir(path):
    """表記ゆれを吸収して実在するフォルダを探す（見つからなければ None）"""
    path = Path(path).expanduser()
    if path.is_dir():
        return path.resolve()
    parts = path.parts
    cur = Path(parts[0]) if path.is_absolute() else Path(".")
    for part in (parts[1:] if path.is_absolute() else parts):
        nxt = cur / part
        if not nxt.is_dir():
            cand = [d for d in cur.iterdir() if d.is_dir() and norm(d.name) == norm(part)] if cur.is_dir() else []
            if not cand:
                return None
            nxt = cand[0]
        cur = nxt
    return cur.resolve()


def load_index(kinds=None):
    """extract_landmarks.py が作った一覧（1行 = 1動画）。kinds で static/dynamic/sokuon を絞る"""
    if not INDEX_CSV.exists():
        raise SystemExit(f"{INDEX_CSV} がありません。先に extract_landmarks.py を実行してください")
    with open(INDEX_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if kinds:
        rows = [r for r in rows if r["kind"] in kinds]
    return rows


def make_groups(rows, regex):
    """ファイル名からグループ（人など）を取り出す。regex が空なら動画ごと"""
    if not regex:
        return np.array([r["video"] for r in rows])
    pat = re.compile(regex)
    out = []
    for r in rows:
        m = pat.search(Path(r["video"]).name)
        out.append((m.group(1) if m.groups() else m.group(0)) if m else "other")
    return np.array(out)


def assign_folds(rows, n_folds=5, group_regex="", seed=0):
    """全動画に fold 番号（0..n-1）を振る。全スクリプトで同じ分割を使うため、同じ入力なら必ず同じ結果になる"""
    from sklearn.model_selection import StratifiedGroupKFold
    labels = np.array([r["label"] for r in rows])
    groups = make_groups(rows, group_regex)
    _, counts = np.unique(labels, return_counts=True)
    n = max(2, min(n_folds, len(set(groups))))
    if counts.min() < n:
        print(f"[注意] 本数が {n} 本未満の文字があります（最少 {counts.min()} 本）。その文字は一部の fold にしか入りません")
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    fold = np.zeros(len(rows), dtype=int)
    for k, (_, test) in enumerate(sgkf.split(np.zeros(len(rows)), labels, groups)):
        fold[test] = k
    return fold, n
