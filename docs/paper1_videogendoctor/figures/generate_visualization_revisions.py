import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "ours": "#0072B2",
    "upper": "#D55E00",
    "baseline": "#666666",
    "open": "#009E73",
    "vlm": "#CC79A7",
    "grid": "#D9D9D9",
}


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(name):
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(path)


def cost_performance_pareto():
    data = [
        ("Rule-only\n+ repair", 0.42, 0.5988, 0.58, COLORS["baseline"]),
        ("Rule+Open-VLM\n+ repair", 0.73, 0.7160, 0.87, COLORS["open"]),
        ("VideoGenDoctor\nfull", 1.00, 0.7639, 1.00, COLORS["ours"]),
        ("Rule+GPT-4V\n+ repair", 2.75, 0.7917, 2.31, COLORS["upper"]),
        ("VLM structured\nreport", 1.16, 0.6810, 1.20, COLORS["vlm"]),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 3.05))
    for label, cost, pass2, latency, color in data:
        size = 95 + 55 * latency
        ax.scatter(cost, pass2, s=size, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        offset = (5, 4)
        if "GPT" in label:
            offset = (-92, -28)
        elif "structured" in label:
            offset = (5, -16)
        elif "Rule-only" in label:
            offset = (5, -12)
        elif "VideoGenDoctor" in label:
            offset = (5, 6)
        ax.annotate(label, (cost, pass2), textcoords="offset points", xytext=offset, ha="left", fontsize=8)

    ax.plot([0.42, 0.73, 1.00, 2.75], [0.5988, 0.7160, 0.7639, 0.7917],
            color="#7A7A7A", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(1.0, color=COLORS["ours"], linewidth=0.8, alpha=0.45)
    ax.axhline(0.7639, color=COLORS["ours"], linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Relative cost (lower is better)")
    ax.set_ylabel("Human Pass@2 (higher is better)")
    ax.set_xlim(0.25, 2.95)
    ax.set_ylim(0.56, 0.82)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax.text(1.04, 0.750, "default operating point", color=COLORS["ours"], fontsize=7)
    ax.text(1.67, 0.806, "higher-cost upper reference", color=COLORS["upper"], fontsize=7)
    fig.tight_layout()
    savefig("cost_performance_pareto")
    plt.close(fig)


def threshold_sensitivity_simulated():
    # Simulated values for planning a future sensitivity experiment.
    # Do not use these numbers as reported experimental results.
    thresholds = np.array([0.30, 0.40, 0.50, 0.60, 0.70])
    macro_f1 = np.array([0.902, 0.910, 0.911, 0.904, 0.886])
    tiou = np.array([0.716, 0.728, 0.732, 0.719, 0.694])
    fpr = np.array([0.158, 0.125, 0.100, 0.083, 0.067])
    unnecessary_patch = np.array([0.142, 0.111, 0.087, 0.071, 0.058])

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(thresholds, macro_f1, marker="o", linewidth=1.8, color="#0072B2", label="Macro-F1")
    ax.plot(thresholds, tiou, marker="s", linewidth=1.8, color="#009E73", label="tIoU@0.5")
    ax2 = ax.twinx()
    ax2.plot(thresholds, fpr, marker="^", linewidth=1.6, color="#D55E00", label="Real-normal FPR")
    ax2.plot(thresholds, unnecessary_patch, marker="D", linewidth=1.6, color="#CC79A7", label="Unnecessary patch")

    ax.axvline(0.50, color="#777777", linewidth=0.9, linestyle="--")
    ax.text(0.505, 0.892, "default", fontsize=8, color="#555555")
    ax.set_xlabel("Diagnosis threshold")
    ax.set_ylabel("Diagnosis / localization metrics")
    ax2.set_ylabel("Risk metrics")
    ax.set_ylim(0.66, 0.93)
    ax2.set_ylim(0.04, 0.18)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout()
    savefig("threshold_sensitivity")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    cost_performance_pareto()
    threshold_sensitivity_simulated()
