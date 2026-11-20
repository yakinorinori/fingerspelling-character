"""
手順1：動画から手の骨格を保存する（時間がかかるのは最初の1回だけ）

入力:  Fingerspelling/Video/transformer訓練/
         静止指文字/かな/動画   → kind=static,  label=かな
         動作指文字/かな/動画   → kind=dynamic, label=かな
         促音/動画               → kind=sokuon,  label=促音
出力:  cache/landmarks/<静止指文字|動作指文字|促音>/.../動画名.npz
         lm    (T,21,3) 画像の正規化座標（x/幅, y/高さ, z）。右手が無いフレームは NaN
         world (T,21,3) world landmarks（手の中心が原点、メートル単位）。カメラからの距離に左右されない
         how   (T,)     右手の決め方（0:右手なし 1:体の手首 2:親指の位置 3:動画全体の親指の位置）
         right (T,)     親指の位置から求めた「右手らしさ」（+ で右手、- で左手。右手が無いフレームは NaN）

右手だけを保存する。右手は次の順で決める（元の動画は左右反転していない前提。反転した動画は --flip）
  1. 体（Pose）の手首が見えている：体の右手首に近い手。ただし親指の位置がはっきり左手を示す手は除く
  2. 体が見えない：親指の位置から右手と判定できる手（右手らしさ ≥ THUMB_MARGIN）
  3. 親指の位置が判定しにくいフレーム（指を平らに伸ばして親指が手のひらの面に近いときなど）：
     手が1つだけで、動画全体で見ると右手なら、その手を右手とする
MediaPipe Hands の Left/Right の判定は使わない（API や撮り方で向きの決まりが変わり、取り違えが起きるため）。

親指の位置による判定（thumb_side）:
  手首→中指の付け根 を u、小指の付け根→人差し指の付け根 を r とすると、u × r は手のひらの面の法線。
  親指の付け根（MCP）は手のひらの面から手のひら側に出ているので、法線と 手首→親指の付け根 の内積の符号で
  左右が決まる（鏡に映すと符号が逆になる）。world landmarks を使うので、手の向きやカメラからの距離に左右されない。
以前の形式で保存したものは、自動で作り直す。
       cache/index.csv（動画の一覧とラベル）

使い方:
    python extract_landmarks.py
    python extract_landmarks.py --train-dir "/Users/.../transformer訓練"
    python extract_landmarks.py --redo
    python extract_landmarks.py --no-pose      # 体を使わない（速い。親指の位置だけで判定する）
"""
import argparse
import csv

import cv2
import numpy as np

from common import (CACHE_DIR, DEFAULT_TRAIN_DIR, GROUP_DIRS, INDEX_CSV, LANDMARK_DIR, SOKUON_LABEL,
                    VIDEO_EXTS, norm, resolve_dir)


POSE_VISIBILITY = 0.5   # 体の手首がこれ以上の確からしさで見えていれば、体から右手を決める
THUMB_MARGIN = 0.15     # 親指の位置による右手らしさが、この値より大きければ右手・-この値より小さければ左手とみなす
HAND_NONE, HAND_BY_POSE, HAND_BY_THUMB, HAND_BY_VIDEO = 0, 1, 2, 3


def thumb_side(world):
    """
    親指の位置から求めた右手らしさ（-1〜+1。+ で右手）。world (21,3)
    普通に撮った画像で右手は負、左手は正になる向きなので、符号を反転して返す
    """
    w = np.asarray(world, dtype=np.float64)
    n = np.cross(w[9] - w[0], w[5] - w[17])
    t = w[2] - w[0]
    return float(-np.dot(n, t) / (np.linalg.norm(n) * np.linalg.norm(t) + 1e-9))


def pose_right_hand(hand_lms, pose_res, flip):
    """体の右手首に最も近く、左手首より右手首に近い手の番号（体が見えなければ None, False）"""
    pose = pose_res.pose_landmarks.landmark if (pose_res and pose_res.pose_landmarks) else None
    if pose is None:
        return None, False
    r_id, l_id = (15, 16) if flip else (16, 15)     # 反転した画像では Pose の「右」は実際の左手
    rw, lw = pose[r_id], pose[l_id]
    if rw.visibility < POSE_VISIBILITY:
        return None, False
    best, best_d = None, np.inf
    for k, h in enumerate(hand_lms):
        w = h.landmark[0]
        d_r = (w.x - rw.x) ** 2 + (w.y - rw.y) ** 2
        d_l = (w.x - lw.x) ** 2 + (w.y - lw.y) ** 2 if lw.visibility >= POSE_VISIBILITY else np.inf
        if d_r < d_l and d_r < best_d:
            best, best_d = k, d_r
    return best, True


def choose_right_hands(frames, margin=THUMB_MARGIN):
    """
    frames: フレームごとの dict（hands: [(lm, world, 右手らしさ)], pose_k: 体から選んだ番号, pose_ok: 体が見えたか）
    戻り値: フレームごとの (選んだ手の番号 or None, 決め方)
    """
    singles = [f["hands"][0][2] for f in frames if len(f["hands"]) == 1 and np.isfinite(f["hands"][0][2])]
    video_right = len(singles) > 0 and np.median(singles) >= margin
    out = []
    for f in frames:
        hs = f["hands"]
        if not hs:
            out.append((None, HAND_NONE))
            continue
        if f["pose_ok"]:
            k = f["pose_k"]
            if k is not None and not (hs[k][2] <= -margin):          # 親指がはっきり左手を示すなら採らない
                out.append((k, HAND_BY_POSE))
            else:
                out.append((None, HAND_NONE))
            continue
        scores = [h[2] for h in hs]
        k = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
        if k is not None and scores[k] >= margin:
            out.append((k, HAND_BY_THUMB))
        elif len(hs) == 1 and video_right and not (scores[0] <= -margin):
            out.append((0, HAND_BY_VIDEO))
        else:
            out.append((None, HAND_NONE))
    return out


def extract_video(video_path, flip=False, use_pose=True):
    """
    動画 → lm (T,21,3), world (T,21,3), 幅, 高さ, fps
    表現者の右手だけを保存する（右手が見つからないフレームは NaN）
    """
    lm, world, width, height, fps, _, _ = extract_video_detail(video_path, flip, use_pose)
    return lm, world, width, height, fps


def extract_video_detail(video_path, flip=False, use_pose=True):
    """extract_video と同じ。加えて右手の決め方 how (T,) と右手らしさ right (T,) を返す"""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise SystemExit("この mediapipe には mp.solutions がありません。既存のコードと同じバージョンの環境で実行してください")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sign = -1.0 if flip else 1.0          # 反転した画像では親指の判定の符号が逆になる
    frames = []
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
            hand_lms = res.multi_hand_landmarks or []
            worlds = res.multi_hand_world_landmarks or []
            hs = []
            for k, h in enumerate(hand_lms):
                lm = [[p.x, p.y, p.z] for p in h.landmark]
                w = [[p.x, p.y, p.z] for p in worlds[k].landmark] if k < len(worlds) else None
                hs.append((lm, w, sign * thumb_side(w) if w is not None else np.nan))
            pose_k, pose_ok = (pose_right_hand(hand_lms, pose.process(rgb), flip)
                               if (pose and hand_lms) else (None, False))
            frames.append({"hands": hs, "pose_k": pose_k, "pose_ok": pose_ok})
    if pose:
        pose.close()
    cap.release()

    T = len(frames)
    lm = np.full((T, 21, 3), np.nan, dtype=np.float32)
    world = np.full((T, 21, 3), np.nan, dtype=np.float32)
    how = np.zeros(T, dtype=np.int8)
    right = np.full(T, np.nan, dtype=np.float32)
    for t, (f, (k, h)) in enumerate(zip(frames, choose_right_hands(frames))):
        how[t] = h
        if k is None:
            continue
        lm[t] = f["hands"][k][0]
        if f["hands"][k][1] is not None:
            world[t] = f["hands"][k][1]
        right[t] = f["hands"][k][2]
    return lm, world, width, height, float(fps), how, right


LANDMARK_VERSION = "right-hand-2"   # 右手だけを保存する形式（親指の位置で判定）。これより前に保存したものは作り直す


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
    parser.add_argument("--no-pose", action="store_true", help="体（Pose）を使わず、親指の位置だけで右手を決める（速い）")
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
                lm, world, w, h, fps, how, right = extract_video_detail(v, args.flip, not args.no_pose)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, lm=lm, world=world, width=w, height=h, fps=fps, how=how, right=right,
                                    version=LANDMARK_VERSION)
                n = max(len(how), 1)
                r = np.nanmedian(right) if np.isfinite(right).any() else np.nan
                print(f"  {label}  {rel.name}: {len(lm)} フレーム, 右手の検出率 {(how > 0).sum() / n:.0%}"
                      f"（体 {(how == HAND_BY_POSE).sum() / n:.0%}, 親指 {(how == HAND_BY_THUMB).sum() / n:.0%}, "
                      f"動画全体 {(how == HAND_BY_VIDEO).sum() / n:.0%}）, 右手らしさ {r:+.2f}")
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
