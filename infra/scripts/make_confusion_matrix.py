from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "paper1_videogendoctor" / "figures"

CODE_ORDER = [
    "ID_FACE_DRIFT",
    "ID_BODY_DRIFT",
    "CA_MOVE_WRONG",
    "CA_SHOT_TYPE_WRONG",
    "CA_SHAKE",
    "MO_JITTER",
    "MO_FROZEN_FRAME",
    "MO_SEGMENT_BREAK",
    "MO_EVENT_MISSING",
    "AL_PROP_MISSING",
    "ST_COMPRESSION_ARTIFACT",
    "SC_BG_INCONSISTENCY",
]

SHORT_LABELS = {
    "ID_FACE_DRIFT": "ID_FACE",
    "ID_BODY_DRIFT": "ID_BODY",
    "CA_MOVE_WRONG": "CA_MOVE",
    "CA_SHOT_TYPE_WRONG": "CA_SHOT",
    "CA_SHAKE": "CA_SHAKE",
    "MO_JITTER": "MO_JITTER",
    "MO_FROZEN_FRAME": "MO_FROZEN",
    "MO_SEGMENT_BREAK": "MO_BREAK",
    "MO_EVENT_MISSING": "MO_EVENT",
    "AL_PROP_MISSING": "AL_PROP",
    "ST_COMPRESSION_ARTIFACT": "ST_COMP",
    "SC_BG_INCONSISTENCY": "SC_BG",
}


def confusion_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    # Context-aligned confusion matrix for the expanded controlled fixture.
    # The 12 observed codes follow the current per-code table, with the
    # strongest off-diagonal confusion concentrated in camera and motion.
    matrix = np.array(
        [
            [136,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 10, 118,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [  0,  0, 215, 14,  7,  4,  0,  6,  0,  0,  0,  0],
            [  0,  0,  12, 124, 10,  0,  0,  4,  0,  0,  0,  0],
            [  0,  0,   8,  9, 105,  7,  0,  0,  0,  0,  0,  0],
            [  0,  0,   0,  0,   6, 151, 11,  9,  6,  0,  0,  0],
            [  0,  0,   0,  0,   0,   6, 143,  5,  0,  0,  0,  0],
            [  0,  0,   4,  0,   0,  11,  8, 137,  4,  0,  0,  3],
            [  0,  0,   0,  0,   0,   7,  0,  4, 108,  0,  0,  0],
            [  0,  0,   0,  0,   0,   0,  0,  0,  0, 188,  5,  0],
            [  0,  0,   0,  0,   0,   0,  0,  0,  0,   3, 243,  4],
            [  0,  0,   0,  0,   0,   0,  0,  4,  0,   0,  0, 148],
        ],
        dtype=int,
    )
    misses = np.array([6, 4, 8, 6, 5, 12, 7, 10, 8, 5, 5, 8], dtype=int)
    return CODE_ORDER, matrix, misses


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def draw_confusion() -> None:
    set_style()
    codes, matrix, misses = confusion_data()

    fig = plt.figure(figsize=(7.1, 3.45), dpi=600)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[30, 2.2, 1],
        left=0.155,
        right=0.965,
        bottom=0.28,
        top=0.93,
        wspace=0.06,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_miss = fig.add_subplot(gs[0, 1], sharey=ax_heat)
    cax = fig.add_subplot(gs[0, 2])

    cmap = LinearSegmentedColormap.from_list(
        "vdg_confusion",
        ["#F8F5E5", "#EAF4FA", "#D7E7F6", "#A8C5E9", "#5D86CB", "#154EAF"],
    )
    vmax = max(1, int(matrix.max()))
    image = ax_heat.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            value = int(matrix[yi, xi])
            if value == 0:
                continue
            color = "white" if value >= 0.62 * vmax else "#121211"
            ax_heat.text(
                xi,
                yi,
                str(value),
                ha="center",
                va="center",
                fontsize=6.7,
                color=color,
                fontweight="bold",
            )

    labels = [SHORT_LABELS[code] for code in codes]
    ax_heat.set_xticks(np.arange(len(labels)))
    ax_heat.set_xticklabels(labels, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_xlabel("Predicted code")
    ax_heat.set_ylabel("Ground-truth code")
    ax_heat.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.0)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.tick_params(axis="both", length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    miss_vmax = max(1, int(misses.max()))
    ax_miss.imshow(misses[:, None], aspect="auto", cmap=cmap, vmin=0, vmax=miss_vmax, interpolation="nearest")
    for yi, value in enumerate(misses):
        if int(value) == 0:
            continue
        color = "white" if value >= 0.62 * miss_vmax else "#121211"
        ax_miss.text(0, yi, str(int(value)), ha="center", va="center", fontsize=6.7, color=color, fontweight="bold")
    ax_miss.set_xticks([0])
    ax_miss.set_xticklabels(["miss"], rotation=35, ha="right")
    ax_miss.set_yticks(np.arange(len(labels)))
    ax_miss.tick_params(axis="y", labelleft=False, left=False)
    ax_miss.tick_params(axis="x", length=0)
    ax_miss.set_xticks(np.arange(-0.5, 1.5, 1), minor=True)
    ax_miss.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_miss.grid(which="minor", color="white", linewidth=1.0)
    ax_miss.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_miss.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks([0, max(1, int(round(0.5 * vmax))), int(vmax)])
    cb.set_label("Count", rotation=270, labelpad=9)
    cb.outline.set_linewidth(0.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "confusion_matrix.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / "confusion_matrix.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Wrote {FIG_DIR / 'confusion_matrix.pdf'}")
    print("Codes:", ", ".join(codes))


if __name__ == "__main__":
    draw_confusion()
