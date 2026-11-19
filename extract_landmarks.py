"""
手順1：動画から手の骨格を保存する（時間がかかるのは最初の1回だけ）

入力:  Fingerspelling/Video/transformer訓練/
         静止指文字/かな/動画   → kind=static,  label=かな
         動作指文字/かな/動画   → kind=dynamic, label=かな
         促音/動画               → kind=sokuon,  label=促音
出力:  cache/landmarks/<静止指文字|動作指文字|促音>/.../動画名.npz
         lm    (T,21,3) 画像の正規化座標（x/幅, y/高さ, z）。手が無いフレームは NaN
         world (T,21,3) world landmarks（手の中心が原点、メートル単位）。カメラからの距離に左右されない
       cache/index.csv（動画の一覧とラベル）

使い方:
    python extract_landmarks.py
    python extract_landmarks.py --train-dir "/Users/.../transformer訓練"
    python extract_landmarks.py --redo
"""
import argparse
import csv

import cv2
import numpy as np

from common import (CACHE_DIR, DEFAULT_TRAIN_DIR, GROUP_DIRS, INDEX_CSV, LANDMARK_DIR, SOKUON_LABEL,
                    VIDEO_EXTS, norm, resolve_dir)


def extract_video(video_path, flip=False):
    """動画 → lm (T,21,3), world (T,21,3), 幅, 高さ, fps"""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise SystemExit("この mediapipe には mp.solutions がありません。既存のコードと同じバージョンの環境で実行してください")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lms, worlds = [], []
    nan = np.full((21, 3), np.nan)
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                  min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lms.append([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark])
                w = res.multi_hand_world_landmarks
                worlds.append([[p.x, p.y, p.z] for p in w[0].landmark] if w else nan)
            else:
                lms.append(nan)
                worlds.append(nan)
    cap.release()
    lm = np.asarray(lms, dtype=np.float32).reshape(-1, 21, 3)
    world = np.asarray(worlds, dtype=np.float32).reshape(-1, 21, 3)
    return lm, world, width, height, float(fps)


def list_videos(folder):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))


def main():
    parser = argparse.ArgumentParser(description="動画から手の骨格を保存する")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="transformer訓練 フォルダ")
    parser.add_argument("--redo", action="store_true", help="保存済みもやり直す")
    parser.add_argument("--flip", action="store_true", help="左右反転してから検出する")
    args = parser.parse_args()

    train_dir = resolve_dir(args.train_dir)
    if train_dir is None:
        raise SystemExit(f"transformer訓練 フォルダが見つかりません: {args.train_dir}（--train-dir で指定できます）")

    rows = []
    for kind, dir_name in GROUP_DIRS.items():
        group_dir = next((d for d in train_dir.iterdir() if d.is_dir() and norm(d.name) == norm(dir_name)), None)
        if group_dir is None:
            print(f"[注意] {dir_name} フォルダがないので飛ばします")
            continue
        videos = list_videos(group_dir)
        print(f"=== {dir_name}: {len(videos)} 本 ===")
        for v in videos:
            rel = v.relative_to(group_dir)
            if kind == "sokuon":
                label = SOKUON_LABEL
            elif len(rel.parts) >= 2:
                label = norm(rel.parts[0])
            else:
                print(f"  [注意] かなのフォルダに入っていないので飛ばします: {rel}")
                continue
            out = LANDMARK_DIR / dir_name / rel.with_suffix(".npz")
            if args.redo or not out.exists():
                lm, world, w, h, fps = extract_video(v, args.flip)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, lm=lm, world=world, width=w, height=h, fps=fps)
                rate = (~np.isnan(lm).any(axis=(1, 2))).mean() if len(lm) else 0
                print(f"  {label}  {rel.name}: {len(lm)} フレーム, 手の検出率 {rate:.0%}")
            rows.append({"npz": str(out.relative_to(CACHE_DIR)), "video": f"{dir_name}/{rel}",
                         "kind": kind, "label": label})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npz", "video", "kind", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 本の一覧を保存しました: {INDEX_CSV}")
    for kind in GROUP_DIRS:
        labels = sorted({r["label"] for r in rows if r["kind"] == kind})
        print(f"  {kind}: {len(labels)} クラス {''.join(labels) if kind != 'sokuon' else ''}")


if __name__ == "__main__":
    main()
