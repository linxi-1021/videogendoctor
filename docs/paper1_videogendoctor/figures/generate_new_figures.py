"""Generate repair-ablation grouped bar chart and controlled-vs-real paired bar chart."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

PALETTE = {
    "ours": "#4F7FAE",
    "ours_light": "#8FB6D9",
    "baseline": "#D7DEE7",
    "baseline_dark": "#7F8B97",
    "highlight": "#E76F51",
    "green": "#6FA675",
    "coral": "#C57E6E",
    "panel": "#FBFCFE",
    "grid": "#E6ECF3",
    "axis": "#7A8794",
    "text": "#28333D",
    "muted": "#677380",
}

def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.8,
        "axes.labelsize": 11.3,
        "axes.titlesize": 11.3,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["axis"],
        "axes.linewidth": 0.85,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.95,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
    })

def soften_axes(ax, y_grid=True):
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.tick_params(colors=PALETTE["text"])
    ax.grid(False)
    if y_grid:
        ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)

def savefig(name):
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        plt.savefig(path)
        print(f"  Saved {path}")


def repair_ablation_bars():
    """Grouped bar chart: Human Pass@2 and Patch Usefulness across ablation conditions."""
    apply_style()

    conditions = [
        "Score-\nonly",
        "Code-\nonly",
        "Code+\nTarget",
        "Code+\nSpan",
        "Code+Span\n+Evidence",
        "VLM-to-\npatch",
        "Patch+\nJudge",
        "Full",
    ]
    human_pass2 = [0.2778, 0.4583, 0.5417, 0.6250, 0.6875, 0.6736, 0.7431, 0.7639]
    patch_useful = [2.28, 2.91, 3.24, 3.57, 3.80, 3.64, 3.93, 4.11]
    new_artifacts = [0.0833, 0.1597, 0.1736, 0.1667, 0.1528, 0.2153, 0.1458, 0.1319]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))

    # Human Pass@2 bars
    bars1 = ax1.bar(x - width, human_pass2, width,
                    color=PALETTE["ours"], edgecolor="white", linewidth=0.6,
                    label="Human Pass@2")
    # Patch usefulness bars (scaled to 0-1 for dual axis)
    usefulness_scaled = [v / 5.0 for v in patch_useful]
    bars2 = ax1.bar(x, usefulness_scaled, width,
                    color=PALETTE["ours_light"], edgecolor="white", linewidth=0.6,
                    label="Patch usefulness (scaled 0-1, raw: 1-5)")

    ax1.set_ylabel("Human Pass@2 / Patch Usefulness (scaled)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=8.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.2, color=PALETTE["text"])
    for i, bar in enumerate(bars2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                 f"{patch_useful[i]:.1f}", ha="center", va="bottom", fontsize=7.2, color=PALETTE["muted"])

    soften_axes(ax1)
    ax1.legend(loc="upper left", fontsize=8.5)

    # New-artifact inset as text annotations
    for i, (cond, na) in enumerate(zip(conditions, new_artifacts)):
        ax1.text(i, -0.07, f"Art:{na:.3f}", ha="center", va="top",
                 fontsize=6.5, color=PALETTE["coral"], style="italic")

    ax1.set_title("Repair Ablation: Additive Contribution of Diagnosis Components", pad=14)
    fig.tight_layout()
    savefig("repair_ablation_bars")
    plt.close()
    print("  repair_ablation_bars done.")


def controlled_vs_real_paired():
    """Paired bar chart: controlled vs real-failure across 4 metrics."""
    apply_style()

    metrics = ["Macro-F1", "tIoU@0.5", "Patch\nUsefulness", "Human\nPass@2"]
    controlled_vlm = [0.8869, 0.6639, 3.62, 0.6810]
    controlled_vgd = [0.9110, 0.7316, 4.11, 0.7639]
    real_vlm = [0.7992, 0.5431, 3.36, 0.6075]
    real_vgd = [0.8264, 0.5982, 3.82, 0.6908]

    x = np.arange(len(metrics))
    width = 0.20

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    bars1 = ax.bar(x - 1.5 * width, controlled_vlm, width,
                   color=PALETTE["baseline"], edgecolor="white", linewidth=0.6,
                   label="VLM struct. (Controlled)")
    bars2 = ax.bar(x - 0.5 * width, controlled_vgd, width,
                   color=PALETTE["ours"], edgecolor="white", linewidth=0.6,
                   label="VideoGenDoctor (Controlled)")
    bars3 = ax.bar(x + 0.5 * width, real_vlm, width,
                   color=PALETTE["baseline"], edgecolor="white", linewidth=0.6,
                   hatch="///", alpha=0.85,
                   label="VLM struct. (Real)")
    bars4 = ax.bar(x + 1.5 * width, real_vgd, width,
                   color=PALETTE["ours"], edgecolor="white", linewidth=0.6,
                   hatch="\\\\", alpha=0.85,
                   label="VideoGenDoctor (Real)")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max(max(controlled_vgd), max(controlled_vlm)) * 1.18)

    # Annotate
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}" if h > 1.0 else f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color=PALETTE["text"], rotation=90)

    soften_axes(ax)
    ax.legend(loc="upper right", fontsize=8.0, ncol=2)

    # Degradation arrows
    for i in range(4):
        mid_ctrl = (controlled_vlm[i] + controlled_vgd[i]) / 2
        mid_real = (real_vlm[i] + real_vgd[i]) / 2
        delta = mid_ctrl - mid_real
        if delta > 0.03:
            ax.annotate(f"{-delta:.2f}", xy=(x[i], mid_real),
                        xytext=(x[i] + 0.35, mid_real - 0.08),
                        fontsize=7.0, color=PALETTE["coral"],
                        arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=0.8))

    ax.set_title("Controlled-Fixture vs. Real-Generator Transfer", pad=14)
    fig.tight_layout()
    savefig("controlled_vs_real_paired")
    plt.close()
    print("  controlled_vs_real_paired done.")


def feature_ablation_horizontal():
    """Horizontal bar chart showing per-feature ablation impact."""
    apply_style()

    features = [
        "Full (all signals)",
        "$-$ Object detection",
        "$-$ Face embedding",
        "$-$ CLIP embedding",
        "$-$ Optical flow",
        "CLIP only",
        "Flow only",
        "Face only",
    ]
    macro_f1 = [0.9110, 0.893, 0.876, 0.871, 0.849, 0.762, 0.638, 0.524]
    tIoU = [0.7316, 0.708, 0.695, 0.683, 0.651, 0.581, 0.472, 0.385]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    y = np.arange(len(features))
    height = 0.30

    bars1 = ax.barh(y + height / 2, macro_f1, height,
                    color=PALETTE["ours"], edgecolor="white", linewidth=0.6,
                    label="Macro-F1")
    bars2 = ax.barh(y - height / 2, tIoU, height,
                    color=PALETTE["ours_light"], edgecolor="white", linewidth=0.6,
                    label="tIoU@0.5")

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9.0)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.invert_yaxis()

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", ha="left", va="center", fontsize=7.2, color=PALETTE["text"])
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", ha="left", va="center", fontsize=7.2, color=PALETTE["muted"])

    soften_axes(ax)
    ax.legend(loc="lower right", fontsize=9.0)
    ax.set_title("Per-Feature Ablation: Signal Contribution to Diagnosis", pad=14)
    fig.tight_layout()
    savefig("feature_ablation_horizontal")
    plt.close()
    print("  feature_ablation_horizontal done.")


if __name__ == "__main__":
    print("Generating new figures...")
    repair_ablation_bars()
    controlled_vs_real_paired()
    feature_ablation_horizontal()
    print("All figures generated.")
