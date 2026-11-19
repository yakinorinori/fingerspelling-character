"""
促音を1クラスとして入れてよいかの確認：手前・奥（z方向）の動きが、骨格から信頼して取れているか

促音（小さい文字・を）は「手を体の方へ引く」動き＝カメラから遠ざかる動き。
カメラから遠ざかると、1メートルあたりの画素数（pixels_per_meter）が小さくなる。
そこで、動画の最初と最後の画素数の比 log(終わり / 始め) を「奥行きの変化」として、
促音と、それ以外の動作指文字・静止指文字でどれだけ分かれるかを見る。

判断の目安（AUC：促音とそれ以外を、この1つの値だけでどれだけ区別できるか。0.5 は区別できない、1.0 は完全）:
  AUC 0.8 以上   → z方向は信頼できる。train_sequence.py に --sokuon を付けて促音を入れる
  AUC 0.65〜0.8 → 手がかりにはなる。--sokuon あり・なしの両方で比べる
  AUC 0.65 未満  → 今の骨格からは z方向を信頼できない。促音は入れない

使い方:
    python check_depth.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import load_index
from dataset import load_raw
from features import pixels_per_meter


def depth_change(row, edge=5):
    lm, world, w, h, _ = load_raw(row)
    ppm = pixels_per_meter(lm, world, w, h)
    ok = np.flatnonzero(np.isfinite(ppm))
    if len(ok) < 2 * edge:
        return np.nan, len(ok) / max(len(ppm), 1)
    start, end = np.median(ppm[ok[:edge]]), np.median(ppm[ok[-edge:]])
    return float(np.log(end / start)), len(ok) / len(ppm)


def main():
    rows = load_index()
    if not any(r["kind"] == "sokuon" for r in rows):
        raise SystemExit("促音の動画がありません（transformer訓練/促音/ を確認してください）")
    vals = {"sokuon": [], "dynamic": [], "static": []}
    rates = []
    for r in rows:
        d, rate = depth_change(r)
        rates.append(rate)
        if np.isfinite(d):
            vals[r["kind"]].append(d)
    print(f"奥行きを測れたフレームの割合（平均）: {np.mean(rates):.0%}\n")
    print("奥行きの変化 log(終わり/始め)（負 = カメラから遠ざかった）")
    for k, name in [("sokuon", "促音"), ("dynamic", "動作指文字"), ("static", "静止指文字")]:
        v = np.array(vals[k])
        if len(v):
            print(f"  {name:6s}: 平均 {v.mean():+.3f}, 標準偏差 {v.std():.3f}, 中央値 {np.median(v):+.3f}（{len(v)} 本）")

    s, others = np.array(vals["sokuon"]), np.array(vals["dynamic"] + vals["static"])
    if len(s) and len(others):
        auc = roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(others))], -np.r_[s, others])
        auc_any = max(auc, 1 - auc)
        print(f"\nAUC（促音 と それ以外）: {auc_any:.3f}"
              + ("" if auc >= 0.5 else "  ※ 促音のほうがカメラに近づいている。撮り方を確認してください"))
        if auc_any >= 0.8:
            print("→ z方向は信頼できます。train_sequence.py に --sokuon を付けて促音を入れてください")
        elif auc_any >= 0.65:
            print("→ 手がかりにはなります。--sokuon あり・なしの両方で比べてください")
        else:
            print("→ 今の骨格からは z方向を信頼できません。促音は入れないほうがよいです")


if __name__ == "__main__":
    main()
