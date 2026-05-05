"""Generate Figure 3 for Paper 1 using the current manuscript numbers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "paper1_videogendoctor" / "figures"


METHOD_SPECS = [
    ("Score-only", 24.07, 35.80, 18.75, 27.78, "#68717D"),
    ("Patch-only", 52.47, 69.75, 47.92, 63.19, "#4C78A8"),
    ("VLM-to-patch", 56.17, 74.38, 50.69, 67.36, "#7A6FF0"),
    ("Patch+Judge", 62.96, 81.79, 56.25, 74.31, "#C85A54"),
    ("VideoGenDoctor-full", 64.81, 83.64, 58.33, 76.39, "#5A9367"),
]


def load_methods() -> list[tuple[str, float, float, float, float, str]]:
    return [(label, auto1, auto2, human1, human2, color) for label, auto1, auto2, human1, human2, color in METHOD_SPECS]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#2F3337",
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.0,
            "xtick.major.size": 2.4,
            "ytick.major.size": 0.0,
            "xtick.color": "#2F3337",
            "ytick.color": "#111827",
            "axes.labelcolor": "#111827",
        }
    )


def draw_panel(ax, methods, idx1: int, idx2: int, title: str, xlabel: str, xmin: float, xmax: float, gain_x: float) -> None:
    y = np.arange(len(methods))
    for yi in y:
        ax.axhline(yi, color="#F5F6F7", linewidth=0.45, zorder=0)

    for yi, method in zip(y, methods):
        label = method[0]
        val1 = method[idx1]
        val2 = method[idx2]
        color = method[5]
        gain = val2 - val1
        ax.plot([val1, val2], [yi, yi], color=color, linewidth=1.05, alpha=0.70, solid_capstyle="round", zorder=2)
        ax.scatter(val1, yi, s=21, facecolor="white", edgecolor=color, linewidth=1.0, marker="o", zorder=3)
        ax.scatter(val2, yi, s=23, facecolor=color, edgecolor=color, linewidth=0.65, marker="o", zorder=4)
        ax.text(val1 - 1.0, yi + 0.12, f"{val1:.1f}", ha="right", va="center", fontsize=5.7, color="#4B5563")
        ax.text(val2 + 0.95, yi + 0.12, f"{val2:.1f}", ha="left", va="center", fontsize=5.7, color="#111827")
        ax.text(gain_x, yi, f"+{gain:.1f}", ha="right", va="center", fontsize=5.6, color="#6B7280")

    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in methods])
    for tick_label, method in zip(ax.get_yticklabels(), methods):
        tick_label.set_color(method[5])
        tick_label.set_fontweight("normal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.52, len(methods) - 0.38)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.42)
    ax.grid(axis="y", visible=False)
    ax.axvline(gain_x - 7.0, color="#EFF1F3", linewidth=0.5, zorder=0)
    ax.text(gain_x, len(methods) - 0.50, "Gain", ha="right", va="bottom", fontsize=5.7, color="#4B5563")
    ax.set_title(title, fontsize=7.0, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def draw() -> None:
    set_style()
    methods = load_methods()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), dpi=600, sharey=True)
    fig.subplots_adjust(left=0.23, right=0.985, bottom=0.22, top=0.78, wspace=0.16)

    draw_panel(
        axes[0],
        methods,
        1,
        2,
        "(a) Automatic verifier outcomes",
        "Automatic pass rate (%)",
        18,
        96,
        95.0,
    )
    draw_panel(
        axes[1],
        methods,
        3,
        4,
        "(b) Blind human outcomes",
        "Human pass rate (%)",
        14,
        90,
        89.0,
    )
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    axes[0].scatter([], [], s=21, facecolor="white", edgecolor="#111827", linewidth=1.0, label="Pass@1")
    axes[0].scatter([], [], s=23, facecolor="#111827", edgecolor="#111827", linewidth=0.65, label="Pass@2")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(1.10, 1.18),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
        borderaxespad=0.0,
        fontsize=6.1,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "manuscript_fig3.png", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG_DIR / "manuscript_fig3.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    draw()
