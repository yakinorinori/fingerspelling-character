"""混同行列の描画（日本語のラベルを表示できるフォントを探して使う）"""
import numpy as np


def _setup_font():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "IPAexGothic", "Noto Sans CJK JP", "Yu Gothic"]:
        if cand in names:
            rcParams["font.family"] = cand
            return True
    return False


def plot_confusion(true, pred, classes, title, path):
    import matplotlib
    has_jp = _setup_font()
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true, pred, labels=range(len(classes)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(classes)
    size = max(5, 0.32 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = classes if has_jp else [str(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("予測" if has_jp else "Predicted")
    ax.set_ylabel("正解" if has_jp else "True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm
