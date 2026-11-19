"""
指の長さの保存と、指がカメラを向いたときの補正（world landmarks 上で行う）

考え方:
  - 骨（関節と関節の間）は指ごとに4本、計20本。長さを骨ごとに保存する
  - 保存の更新は「骨が画面とほぼ平行なフレーム」（|dz| / 長さ < UPDATE_Z）だけで行い、最大値で更新する
      ・最初に手が見えたフレームの長さで始め、信頼できるフレームが来たらその値に置き換える
      ・信頼できる値が入った後は、MAX_GROWTH 倍を超える更新は検出ミスとみなして無視する
  - 骨が強くカメラを向いているフレーム（|dz| / 長さ > CORRECT_Z）では、保存した長さで骨を作り直す
      ・x, y は検出値のまま、z = 符号 × √(L² − x² − y²)
  - 手が RESET_FRAMES フレーム続けて消えたら、別の場面とみなして保存した長さをリセットする
world landmarks はメートル単位で手の中心が原点なので、後ろに下がっても長さは変わらない。
"""
import numpy as np

FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
BONES = [(f[i], f[i + 1]) for f in FINGERS for i in range(4)]   # 20本。指 f の i 本目 = 4f+i
PARENT = np.array([p for p, _ in BONES])
CHILD = np.array([c for _, c in BONES])

UPDATE_Z = 0.3
CORRECT_Z = 0.7
MAX_GROWTH = 1.3
RESET_FRAMES = 30


def correct_sequence(world, update_z=UPDATE_Z, correct_z=CORRECT_Z, max_growth=MAX_GROWTH,
                     reset_frames=RESET_FRAMES):
    """
    world (T,21,3), 手が無いフレームは NaN
    戻り値:
      corrected (T,21,3)  補正後の骨格（手首が原点。手が無いフレームは NaN）
      finger_flag (T,5)   その指の骨を1本でも補正したら 1
      lengths (T,20)      そのフレームで使った保存済みの長さ
    """
    T = len(world)
    corrected = np.full((T, 21, 3), np.nan, dtype=np.float32)
    finger_flag = np.zeros((T, 5), dtype=np.float32)
    lengths = np.full((T, 20), np.nan, dtype=np.float32)
    stored = np.full(20, np.nan)
    reliable = np.zeros(20, dtype=bool)
    missing = 0
    eps = 1e-9

    for t in range(T):
        w = world[t]
        if np.isnan(w).any():
            missing += 1
            if missing >= reset_frames:
                stored[:] = np.nan
                reliable[:] = False
            continue
        missing = 0
        vec = (w[CHILD] - w[PARENT]).astype(np.float64)        # (20,3)
        length = np.linalg.norm(vec, axis=1)
        zr = np.abs(vec[:, 2]) / (length + eps)
        good = zr < update_z

        # --- 保存した長さの更新 ---
        first = np.isnan(stored)
        stored[first] = length[first]
        promote = good & ~reliable                               # 初めて信頼できる値が来た骨は置き換える
        stored[promote] = length[promote]
        reliable |= promote
        grow = good & reliable & ~promote & (length > stored) & (length <= max_growth * stored)
        stored[grow] = length[grow]

        # --- カメラを向いた骨の補正 ---
        fix = (zr > correct_z) & reliable
        if fix.any():
            L = stored[fix]
            xy = vec[fix, :2]
            r2 = (xy ** 2).sum(1)
            over = r2 >= L ** 2                                  # x,y だけで既に長い → 長さに合わせて縮め、z=0
            scale = np.where(over, L / np.sqrt(r2 + eps), 1.0)
            xy = xy * scale[:, None]
            dz = np.where(over, 0.0, np.sign(vec[fix, 2] + eps) * np.sqrt(np.maximum(L ** 2 - r2, 0)))
            vec[fix] = np.column_stack([xy, dz])
            finger_flag[t] = [fix[4 * f:4 * f + 4].any() for f in range(5)]

        # --- 手首から順に骨格を組み直す ---
        pos = np.zeros((21, 3))
        for b, (p, c) in enumerate(BONES):
            pos[c] = pos[p] + vec[b]
        corrected[t] = pos
        lengths[t] = stored
    return corrected, finger_flag, lengths
