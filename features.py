"""
特徴量（保存した骨格から計算する）

ブロック（1フレームごと）:
  H1  手型・既存   30  deta.py の hand_data と同じ計算（不具合も含めてそのまま。過去の結果との比較用）
  H3  手型・新規   43  指の長さで補正した world 骨格から：関節の曲がり15、第二関節から先の曲がり5、
                       指先どうしの距離10、指の開き（付け根4・第二関節4）、補正フラグ5
  O   向き          6  手のひらの法線3、手首→中指の付け根の方向3（補正後の world 骨格）
  M   動き         18  手のひら中心の動きを前後7フレームの窓でまとめたもの（移動量は「実際の距離」に換算）
  S   SVMの確率    41  静止SVMの出力（train_static_svm.py が作る。ここでは計算しない）

特徴量セットは "H3+O+M" のようにブロックを + でつなげて指定する。
"""
import math

import numpy as np

from skeleton import BONES, FINGERS, correct_sequence

BLOCK_DIMS = {"H1": 30, "H3": 43, "O": 6, "M": 18}
PALM_IDS = [0, 5, 9, 13, 17]
EPS = 1e-9


# ---------------------------------------------------------------------------
# H1：既存の30次元（deta.py / Video_test.py の hand_data と同じ）
# ---------------------------------------------------------------------------
def _h1_frame(L, w, h):
    P = lambda i: (int(L[i, 0] * w), int(L[i, 1] * h))
    d2 = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    WR, T_MCP, I_MCP, M_MCP, R_MCP, P_MCP = P(0), P(1), P(5), P(9), P(13), P(17)
    T_IP, T_TIP = P(3), P(4)
    I_PIP, I_TIP, M_PIP, M_TIP = P(7), P(8), P(11), P(12)
    R_PIP, R_TIP, Pk_PIP, Pk_TIP = P(15), P(16), P(19), P(20)
    R1, R2, R3, R4 = d2(WR, T_MCP), d2(WR, I_MCP), d2(WR, M_MCP), d2(WR, R_MCP)
    R5 = (WR[0] - I_MCP[0]) ** 2 + (WR[1] - P_MCP[1]) ** 2      # 元のコードのまま（人差し指の x と小指の y）
    RY = d2(I_MCP, P_MCP)
    R1, R2, R3, R4, R5, RY = [r if r != 0 else EPS for r in (R1, R2, R3, R4, R5, RY)]
    JT1, JT2 = d2(WR, T_IP), d2(WR, T_TIP)
    JI1, JI2 = d2(WR, I_PIP), d2(WR, I_TIP)
    JM1, JM2 = d2(WR, M_PIP), d2(WR, M_TIP)
    JR1, JR2 = d2(WR, R_PIP), d2(WR, R_TIP)
    JP1, JP2 = d2(WR, Pk_PIP), d2(WR, Pk_TIP)
    f = [JT1 / R1, JT1 / R1 - JT2 / R1, JI1 / R2, JI1 / R2 - JI2 / R2, JM1 / R3, JM1 / R3 - JM2 / R3,
         JR1 / R4, JR1 / R4 - JR2 / R4, JP1 / R5 * 2, (JP1 / R5 - JP2 / R5) * 2,
         d2(T_TIP, I_TIP) / RY, d2(T_TIP, M_TIP) / RY, d2(T_TIP, R_TIP) / RY, d2(T_TIP, Pk_TIP) / RY,
         d2(I_TIP, M_TIP) / RY, d2(M_TIP, R_TIP) / RY, d2(R_TIP, Pk_TIP) / RY]
    X = lambda i: L[i, 0] * w
    Y = lambda i: L[i, 1] * h
    Z = lambda i: L[i, 2] * w
    ang = lambda a, b: np.arctan2(X(a) - X(b), Y(a) - Y(b))
    f += [ang(9, 0),
          np.arctan2(X(4) - X(0), Z(4) - Z(0)),
          np.arctan2(X(20) - X(0), Z(20) - Z(0))]
    f += [ang(4, 9), ang(8, 9), ang(12, 9), ang(16, 9), ang(20, 9),
          ang(4, 3), ang(12, 10), ang(20, 18), ang(8, 6), ang(16, 14)]
    return f


def h1_features(lm, width, height):
    out = np.zeros((len(lm), 30), dtype=np.float32)
    for t in range(len(lm)):
        if not np.isnan(lm[t]).any():
            out[t] = _h1_frame(lm[t], width, height)
    return out


# ---------------------------------------------------------------------------
# H3：補正後の world 骨格から計算する手型（手の向き・カメラからの距離に左右されない）
# ---------------------------------------------------------------------------
TIPS = [4, 8, 12, 16, 20]
TIP_PAIRS = [(a, b) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]]          # 10組


def _angle(u, v):
    return math.acos(float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + EPS), -1, 1)))


def h3_features(corr, flags, lengths):
    """
    1フレーム43次元:
      bend      15  各関節の曲がり（隣り合う骨のなす角。付け根・第二関節・第一関節 × 5指）
      pip_curl   5  第二関節から先の曲がり：「付け根→第二関節」と「第二関節→指先」のなす角
                    （第一関節・指先の小さな曲がりも足し合わさるので、少し曲がった指が区別しやすい）
      dist      10  指先どうしの距離（手の長さで割る）
      spread     4  隣の指との開き（付け根→第二関節の向きどうし）
      spread_pip 4  隣の指との開き（第二関節→指先の向きどうし。指先側がそろっているか・離れているか）
      flags      5  その指を補正したか
    親指は「付け根」= CMC(1), 「第二関節」= MCP(2) として同じ計算をする
    """
    out = np.zeros((len(corr), 43), dtype=np.float32)
    for t in range(len(corr)):
        p = corr[t]
        if np.isnan(p).any():
            continue
        vec = [[p[f[i + 1]] - p[f[i]] for i in range(4)] for f in FINGERS]
        bend = [_angle(vec[f][i], vec[f][i + 1]) for f in range(5) for i in range(3)]      # 15
        distal = [p[f[4]] - p[f[2]] for f in FINGERS]                                     # 第二関節→指先
        pip_curl = [_angle(vec[f][1], distal[f]) for f in range(5)]                       # 5
        hand_len = lengths[t, 8] if np.isfinite(lengths[t, 8]) else np.linalg.norm(p[9] - p[0])  # 手首→中指の付け根
        dist = [np.linalg.norm(p[a] - p[b]) / (hand_len + EPS) for a, b in TIP_PAIRS]      # 10
        spread = [_angle(vec[f][1], vec[f + 1][1]) for f in range(4)]                     # 4
        spread_pip = [_angle(distal[f], distal[f + 1]) for f in range(4)]                 # 4
        out[t] = bend + pip_curl + dist + spread + spread_pip + list(flags[t])            # +5
    return out


# ---------------------------------------------------------------------------
# O：向き
# ---------------------------------------------------------------------------
def o_features(corr):
    out = np.zeros((len(corr), 6), dtype=np.float32)
    for t in range(len(corr)):
        p = corr[t]
        if np.isnan(p).any():
            continue
        n = np.cross(p[5] - p[0], p[17] - p[0])
        d = p[9] - p[0]
        out[t, :3] = n / (np.linalg.norm(n) + EPS)
        out[t, 3:] = d / (np.linalg.norm(d) + EPS)
    return out


# ---------------------------------------------------------------------------
# M：動き（動作モデルの案Aを、移動量の単位を「実際の距離」に変えたもの）
# ---------------------------------------------------------------------------
M_NAMES = ([f"dir8_{d}" for d in ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]]
           + ["hv_ratio", "consistency_x", "consistency_y", "straightness",
              "pca_cos2", "pca_sin2", "linearity", "rotation", "depth_change", "path_length"])


def _interp(values, valid):
    t = np.arange(len(values))
    out = np.empty_like(values, dtype=np.float64)
    for d in range(values.shape[1]):
        out[:, d] = np.interp(t, t[valid], values[valid, d])
    return out


def pixels_per_meter(lm, world, width, height, update_z=0.3):
    """
    1メートルあたりの画素数（フレームごと）。画面と平行な骨（|dz|/長さ < update_z）で
    「画像上の長さ ÷ world の長さ」を求め、その中央値を使う。後ろに下がると小さくなる。
    """
    T = len(lm)
    ppm = np.full(T, np.nan)
    from skeleton import CHILD, PARENT
    for t in range(T):
        if np.isnan(lm[t]).any() or np.isnan(world[t]).any():
            continue
        wv = world[t, CHILD] - world[t, PARENT]
        wl = np.linalg.norm(wv, axis=1)
        ok = (np.abs(wv[:, 2]) / (wl + EPS) < update_z) & (wl > 0.005)
        if ok.sum() < 3:
            continue
        iv = (lm[t, CHILD, :2] - lm[t, PARENT, :2]) * np.array([width, height])
        ppm[t] = np.median(np.linalg.norm(iv[ok], axis=1) / wl[ok])
    return ppm


def m_features(lm, world, width, height, half_window=7):
    T = len(lm)
    out = np.zeros((T, 18), dtype=np.float32)
    valid = ~np.isnan(lm).any(axis=(1, 2))
    if valid.sum() < 2:
        return out
    ppm = pixels_per_meter(lm, world, width, height)
    ok = np.isfinite(ppm)
    if ok.sum() == 0:                      # world が取れていない場合は、画像上の手の大きさで代用
        size = np.linalg.norm((lm[:, 9, :2] - lm[:, 0, :2]) * [width, height], axis=1)
        ppm, ok = size / 0.09, valid       # 手首→中指の付け根 ≒ 9cm と仮定
    ppm = _interp(ppm[:, None], ok)[:, 0]
    # 中央値フィルタ（5フレーム）でぶれを抑える
    ppm = np.array([np.median(ppm[max(0, t - 2):t + 3]) for t in range(T)])

    xy = lm[:, :, :2] * np.array([width, height], dtype=np.float64)
    center = _interp(xy[:, PALM_IDS].mean(axis=1), valid)                   # 画素
    vel = np.vstack([[0, 0], np.diff(center, axis=0)]) / ppm[:, None]            # メートル/フレーム
    vel[:, 1] *= -1                                                              # 上を正に
    traj = np.cumsum(vel, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    theta = np.arctan2(vel[:, 1], vel[:, 0])
    bins = (np.floor((theta + np.pi / 8) / (np.pi / 4)).astype(int)) % 8
    hand_vec = _interp(xy[:, 9] - xy[:, 0], valid)
    angle = np.unwrap(np.arctan2(hand_vec[:, 1], hand_vec[:, 0]))
    d_angle = np.concatenate([[0], np.diff(angle)])
    d_depth = np.concatenate([[0], np.diff(np.log(ppm + EPS))])                  # 手前に来ると正

    for t in range(T):
        a, b = max(0, t - half_window), min(T, t + half_window + 1)
        v, s = vel[a:b], speed[a:b]
        hist = np.bincount(bins[a:b], weights=s, minlength=8)
        path = s.sum()
        if path > 1e-6:
            sx2, sy2 = (v[:, 0] ** 2).sum(), (v[:, 1] ** 2).sum()
            hv = sx2 / (sx2 + sy2 + EPS)
            cx = v[:, 0].sum() / (np.abs(v[:, 0]).sum() + EPS)
            cy = v[:, 1].sum() / (np.abs(v[:, 1]).sum() + EPS)
            straight = np.linalg.norm(v.sum(0)) / (path + EPS)
        else:
            hv, cx, cy, straight = 0.5, 0.0, 0.0, 0.0
        p = traj[a:b]
        w_, vecs = np.linalg.eigh(np.cov(p.T)) if len(p) > 1 else (np.zeros(2), np.eye(2))
        if w_.sum() > 1e-10:
            phi = math.atan2(vecs[1, 1], vecs[0, 1])
            c2, s2, lin = math.cos(2 * phi), math.sin(2 * phi), w_[1] / (w_.sum() + EPS)
        else:
            c2 = s2 = lin = 0.0
        out[t] = [*hist, hv, cx, cy, straight, c2, s2, lin, d_angle[a:b].sum(), d_depth[a:b].sum(), path]
    return out


# ---------------------------------------------------------------------------
def compute_blocks(lm, world, width, height, need=("H1", "H3", "O", "M")):
    """必要なブロックをまとめて計算する（補正は1回だけ）"""
    blocks = {}
    if "H1" in need:
        blocks["H1"] = h1_features(lm, width, height)
    if "H3" in need or "O" in need:
        corr, flags, lengths = correct_sequence(world)
        if "H3" in need:
            blocks["H3"] = h3_features(corr, flags, lengths)
        if "O" in need:
            blocks["O"] = o_features(corr)
    if "M" in need:
        blocks["M"] = m_features(lm, world, width, height)
    return blocks


def parse_set(feature_set):
    return [b.strip() for b in feature_set.split("+") if b.strip()]


def build(feature_set, blocks, extra=None):
    """'H3+O+M' → ブロックを横につなげた (T,D)。S など外部のブロックは extra で渡す"""
    parts = []
    for b in parse_set(feature_set):
        src = extra if (extra and b in extra) else blocks
        if b not in src:
            raise KeyError(f"特徴量ブロック {b} がありません")
        parts.append(src[b])
    return np.concatenate(parts, axis=1).astype(np.float32)
