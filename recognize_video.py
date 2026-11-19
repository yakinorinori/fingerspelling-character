"""
手順5：実際の動画で認識結果を見る（統一モデル・動作モデル・静止SVM）

  - 統一モデル / 動作モデル（.pt）：動画全体での予測と、窓をずらしながらの予測を表示
  - 静止SVM（.joblib）：フレームごとの予測を表示し、中央の半分の平均で動画全体を予測
ラベル付きの動画は results/recognized/ に保存する。

使い方:
    python recognize_video.py --model "results/models/unified_transformer_S+O+M.pt" 動画.mp4
    python recognize_video.py --model "results/models/svm_H3+O.joblib" フォルダ/
    python recognize_video.py --model ... 動画.mp4 --window 30 --stride 5 --no-show
"""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from common import MODEL_DIR, RESULT_DIR, VIDEO_EXTS
from extract_landmarks import extract_video
from features import build, compute_blocks, parse_set
from models import build_model
from train_static_svm import center_frames

FONT_CANDIDATES = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                   "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]


class Drawer:
    """日本語を描けるフォントがあれば Pillow で、なければ英数字で描く"""
    def __init__(self):
        self.font = None
        try:
            from PIL import ImageFont
            for p in FONT_CANDIDATES:
                if Path(p).exists():
                    self.font = ImageFont.truetype(p, 32)
                    self.small = ImageFont.truetype(p, 20)
                    break
        except ImportError:
            pass

    def text(self, frame, lines):
        if self.font is None:
            for i, (s, color) in enumerate(lines):
                s = s.encode("ascii", "replace").decode()
                cv2.putText(frame, s, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(frame, s, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            return frame
        from PIL import Image, ImageDraw
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for i, (s, color) in enumerate(lines):
            font = self.font if i < 2 else self.small
            d.text((20, 15 + 42 * i), s, font=font, fill=color[::-1], stroke_width=3, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def top(probs, classes, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return "  ".join(f"{classes[i]} {probs[i]:.2f}" for i in idx)


def load(path):
    if str(path).endswith(".joblib"):
        d = joblib.load(path)
        return {"type": "svm", **d}
    d = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(d["model"], d["in_dim"], len(d["classes"]))
    model.load_state_dict(d["state_dict"])
    model.eval()
    d["net"] = model
    if "S" in parse_set(d["features"]):
        svm_path = MODEL_DIR / f"svm_{d['svm_features']}.joblib"
        if not svm_path.exists():
            raise SystemExit(f"統一モデルには {svm_path} が必要です（train_static_svm.py で作られます）")
        d["svm"] = joblib.load(svm_path)
    d["type"] = "seq"
    return d


def svm_probs(svm_pack, blocks, valid):
    svm = svm_pack["svm"]
    feat = build(svm_pack["features"], blocks)
    p = np.zeros((len(feat), len(svm_pack["classes"])), dtype=np.float32)
    if valid.any():
        p[np.ix_(np.flatnonzero(valid), svm.classes_)] = svm.predict_proba(feat[valid])
    return p


def run_seq(pack, seq):
    x = torch.from_numpy(((seq - pack["mean"]) / pack["std"]).astype(np.float32)[None])
    with torch.no_grad():
        return torch.softmax(pack["net"](x, torch.ones(1, len(seq), dtype=torch.bool)), 1)[0].numpy()


def process(video, pack, args, drawer):
    print(f"\n動画: {video}")
    lm, world, w, h, fps = extract_video(video, args.flip)
    if len(lm) == 0:
        print("  フレームが読めませんでした")
        return
    valid = ~np.isnan(lm).any(axis=(1, 2))
    blocks = compute_blocks(lm, world, w, h)
    T = len(lm)
    frame_text = [None] * T

    if pack["type"] == "svm":
        classes = pack["classes"]
        probs = svm_probs(pack, blocks, valid)
        c = center_frames(valid)
        whole = probs[c].mean(0) if len(c) else np.zeros(len(classes))
        for t in range(T):
            if valid[t]:
                frame_text[t] = "フレーム: " + top(probs[t], classes)
    else:
        classes = pack["classes"]
        extra = {"S": svm_probs(pack["svm"], blocks, valid)} if "svm" in pack else None
        seq = build(pack["features"], blocks, extra)
        whole = run_seq(pack, seq)
        if args.window and T >= args.window:
            starts = np.arange(0, T - args.window + 1, args.stride)
            wins = [run_seq(pack, seq[s:s + args.window]) for s in starts]
            centers = starts + args.window // 2
            for t in range(T):
                frame_text[t] = "窓: " + top(wins[int(np.abs(centers - t).argmin())], classes)
            print(f"  窓ごとの予測: {''.join(classes[int(p.argmax())][0] for p in wins)}")

    print(f"  手の検出率 {valid.mean():.0%}, {T} フレーム")
    print(f"  動画全体の予測: {top(whole, classes, 5)}")

    out_dir = RESULT_DIR / "recognized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(video).stem}_{Path(args.model).stem}_labeled.mp4"
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(T):
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)
        lines = [(f"動画全体: {classes[int(whole.argmax())]} ({whole.max():.2f})", (0, 255, 0))]
        if frame_text[t]:
            lines.append((frame_text[t], (0, 200, 255)))
        if not valid[t]:
            lines.append(("手が見つからない", (0, 0, 255)))
        frame = drawer.text(frame, lines)
        writer.write(frame)
        if not args.no_show:
            cv2.imshow("recognize", frame)
            if cv2.waitKey(int(1000 / fps)) == 27:
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"  保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="動画を入れて認識結果を見る")
    parser.add_argument("inputs", nargs="+", help="動画ファイル、または動画の入ったフォルダ")
    parser.add_argument("--model", required=True, help=".pt（統一・動作）または .joblib（静止SVM）")
    parser.add_argument("--window", type=int, default=30, help="窓の長さ（0 で窓ごとの予測をしない）")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--flip", action="store_true", help="左右反転してから検出する（骨格保存のときと合わせる）")
    parser.add_argument("--no-show", action="store_true", help="画面に表示せず保存だけする")
    args = parser.parse_args()

    pack = load(args.model)
    drawer = Drawer()
    videos = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            videos += sorted(v for v in p.rglob("*") if v.suffix.lower() in VIDEO_EXTS and not v.name.startswith("."))
        elif p.is_file():
            videos.append(p)
        else:
            print(f"見つかりません: {p}")
    for v in videos:
        process(v, pack, args, drawer)


if __name__ == "__main__":
    main()
