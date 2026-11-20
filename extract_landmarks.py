"""
手順1：動画から手の骨格を保存する（時間がかかるのは最初の1回だけ）

入力:  Fingerspelling/Video/transformer訓練/
         静止指文字/かな/動画   → kind=static,  label=かな
         動作指文字/かな/動画   → kind=dynamic, label=かな
         促音/動画               → kind=sokuon,  label=促音
出力:  cache/landmarks/<静止指文字|動作指文字|促音>/.../動画名.npz
         lm    (T,21,3) 画像の正規化座標（x/幅, y/高さ, z）。右手が無いフレームは NaN
         world (T,21,3) world landmarks（手の中心が原点、メートル単位）。カメラからの距離に左右されない
         how   (T,)     右手の決め方（0:右手なし 1:体の手首から 2:Hands の左右判定から）

右手だけを保存する。体（Pose）が映っていれば体の右手首に近い手を右手とし、
映っていなければ MediaPipe Hands の左右判定を使う（元の動画は左右反転していない前提）。
以前の形式（右手を区別していない）で保存したものは、自動で作り直す。
       cache/index.csv（動画の一覧とラベル）

使い方:
    python extract_landmarks.py
    python extract_landmarks.py --train-dir "/Users/.../transformer訓練"
    python extract_landmarks.py --redo
    python extract_landmarks.py --no-pose      # 体を使わない（速いが、左右判定を誤ることがある）
"""
import argparse
import csv

import cv2
import numpy as np

from common import (CACHE_DIR, DEFAULT_TRAIN_DIR, GROUP_DIRS, INDEX_CSV, LANDMARK_DIR, SOKUON_LABEL,
                    VIDEO_EXTS, norm, resolve_dir)


POSE_VISIBILITY = 0.5            # 体の手首がこれ以上の確からしさで見えていれば、体から右手を決める
HAND_NONE, HAND_BY_POSE, HAND_BY_LABEL = 0, 1, 2


def pick_right_hand(res, pose_res, flip):
    """
    検出した手（最大2つ）から、表現者の右手を選ぶ。戻り値: (手の番号 or None, 決め方)

    1. 体（Pose）の手首が見えているとき：体の右手首に最も近く、かつ左手首より右手首に近い手
    2. 体が見えないとき：MediaPipe Hands の左右の判定
         Hands は「鏡のように左右反転した画像」を前提に左右を返すので、普通に撮った動画では
         "Left" が表現者の右手。--flip で反転して入力したときは "Right" が右手
    ※ 元の動画が普通のカメラで撮ったもの（左右反転していない）とする
    """
    hands = res.multi_hand_landmarks or []
    if not hands:
        return None, HAND_NONE
    pose = pose_res.pose_landmarks.landmark if (pose_res and pose_res.pose_landmarks) else None
    if pose is not None:
        # 反転した画像では、Pose の「右」は実際の左手になる
        r_id, l_id = (15, 16) if flip else (16, 15)
        rw, lw = pose[r_id], pose[l_id]
        if rw.visibility >= POSE_VISIBILITY:
            best, best_d = None, np.inf
            for k, h in enumerate(hands):
                w = h.landmark[0]
                d_r = (w.x - rw.x) ** 2 + (w.y - rw.y) ** 2
                d_l = (w.x - lw.x) ** 2 + (w.y - lw.y) ** 2 if lw.visibility >= POSE_VISIBILITY else np.inf
                if d_r < d_l and d_r < best_d:
                    best, best_d = k, d_r
            return best, (HAND_BY_POSE if best is not None else HAND_NONE)
    right_label = "Right" if flip else "Left"
    for k, hd in enumerate(res.multi_handedness or []):
        if hd.classification[0].label == right_label:
            return k, HAND_BY_LABEL
    return None, HAND_NONE


def extract_video(video_path, flip=False, use_pose=True):
    """
    動画 → lm (T,21,3), world (T,21,3), 幅, 高さ, fps
    表現者の右手だけを保存する（右手が見つからないフレームは NaN）
    """
    lm, world, width, height, fps, _ = extract_video_detail(video_path, flip, use_pose)
    return lm, world, width, height, fps


def extract_video_detail(video_path, flip=False, use_pose=True):
    """extract_video と同じ。加えて各フレームの右手の決め方 how (T,)（0:なし 1:体から 2:左右判定から）を返す"""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise SystemExit("この mediapipe には mp.solutions がありません。既存のコードと同じバージョンの環境で実行してください")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lms, worlds, hows = [], [], []
    nan = np.full((21, 3), np.nan)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5) if use_pose else None
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                  min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            pose_res = pose.process(rgb) if (pose and res.multi_hand_landmarks) else None
            k, how = pick_right_hand(res, pose_res, flip)
            if k is not None:
                lms.append([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[k].landmark])
                w = res.multi_hand_world_landmarks
                worlds.append([[p.x, p.y, p.z] for p in w[k].landmark] if w and len(w) > k else nan)
            else:
                lms.append(nan)
                worlds.append(nan)
            hows.append(how)
    if pose:
        pose.close()
    cap.release()
    lm = np.asarray(lms, dtype=np.float32).reshape(-1, 21, 3)
    world = np.asarray(worlds, dtype=np.float32).reshape(-1, 21, 3)
    return lm, world, width, height, float(fps), np.asarray(hows, dtype=np.int8)


LANDMARK_VERSION = "right-hand-1"   # 右手だけを保存する形式。これより前に保存したものは作り直す


def is_current(path):
    if not path.exists():
        return False
    try:
        d = np.load(path)
        return "version" in d.files and str(d["version"]) == LANDMARK_VERSION
    except Exception:
        return False


def list_videos(folder):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))


def main():
    parser = argparse.ArgumentParser(description="動画から手の骨格を保存する")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="transformer訓練 フォルダ")
    parser.add_argument("--redo", action="store_true", help="保存済みもやり直す")
    parser.add_argument("--flip", action="store_true", help="左右反転してから検出する")
    parser.add_argument("--no-pose", action="store_true", help="体（Pose）を使わず、Hands の左右判定だけで右手を決める（速い）")
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
            if args.redo or not is_current(out):
                lm, world, w, h, fps, how = extract_video_detail(v, args.flip, not args.no_pose)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, lm=lm, world=world, width=w, height=h, fps=fps, how=how,
                                    version=LANDMARK_VERSION)
                n = max(len(how), 1)
                print(f"  {label}  {rel.name}: {len(lm)} フレーム, 右手の検出率 {(how > 0).mean() if len(how) else 0:.0%}"
                      f"（体から {(how == HAND_BY_POSE).sum() / n:.0%}, 左右判定から {(how == HAND_BY_LABEL).sum() / n:.0%}）")
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
