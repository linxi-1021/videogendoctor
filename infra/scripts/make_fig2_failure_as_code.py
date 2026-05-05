from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "paper1_videogendoctor" / "figures"


def real_failure_data():
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
    groups = ["ID", "CA", "MO", "AL", "ST", "SC"]
    matrix = np.array(
        [
            [52, 58, 47, 39, 31, 21],
            [44, 52, 58, 61, 59, 52],
            [46, 58, 71, 82, 88, 85],
            [20, 32, 44, 52, 54, 46],
            [10, 16, 22, 26, 32, 30],
            [16, 16, 20, 26, 22, 48],
        ],
        dtype=float,
    )
    return bins, groups, matrix


def controlled_profile_data():
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
    totals = np.array([332, 406, 448, 372, 270, 188], dtype=float)
    return bins, totals


def draw_bar_panel(fig_name: str, panel_title: str, totals: np.ndarray):
    set_style()
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
    fig, ax = plt.subplots(figsize=(7.1, 1.55), dpi=600)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.84)

    x = np.arange(len(bins))
    ax.bar(x, totals, width=0.78, color="#EAF4FA", edgecolor="none", zorder=1)
    ax.plot(x, totals, color="#145BC6", linewidth=1.55, marker="o", markersize=3.6, zorder=3)
    ax.scatter(x[np.argmax(totals)], totals.max(), s=24, color="#EE6F04", zorder=4)
    for xi, total in zip(x, totals):
        ax.text(xi, total + 2.0, f"{int(total)}", ha="center", va="bottom", fontsize=6.8, color="#121211")
    upper = max(30, int(np.ceil(totals.max() / 5.0) * 5))
    ax.set_ylim(0, upper)
    ax.set_yticks(np.linspace(0, upper, 4, dtype=int))
    ax.set_ylabel("Verified spans")
    ax.set_xlabel("Time bin (seconds)")
    ax.set_title(panel_title, loc="left", fontsize=8.2, pad=2)
    ax.grid(axis="y", color="#E9EAE9", linewidth=0.5)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5F6362")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{fig_name}.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / f"{fig_name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def set_style():
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


def draw_profile():
    _, totals = controlled_profile_data()
    draw_bar_panel("temporal_evidence_profile", "(a) Controlled fixture: 2,016 verified spans", totals)


def draw_real_profile():
    _, _, matrix = real_failure_data()
    col_totals = matrix.sum(axis=0).astype(float)
    draw_bar_panel("temporal_evidence_real_profile", "(b) Real-failure subset: 1,536 verified spans", col_totals)


def draw_heatmap():
    set_style()
    bins, groups, matrix = real_failure_data()
    row_totals = matrix.sum(axis=1).astype(int)
    fig = plt.figure(figsize=(7.1, 2.42), dpi=600)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[30, 1.8, 1],
        left=0.085,
        right=0.965,
        bottom=0.17,
        top=0.84,
        wspace=0.04,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_total = fig.add_subplot(gs[0, 1], sharey=ax_heat)
    cax = fig.add_subplot(gs[0, 2])

    x = np.arange(len(bins))
    cmap = LinearSegmentedColormap.from_list(
        "vdg_overview_blue",
        ["#F8F5E5", "#EAF4FA", "#DFEDF8", "#AFC7ED", "#647ECE", "#145BC6"],
    )

    vmax = max(1, matrix.max())
    image = ax_heat.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            value = int(matrix[yi, xi])
            if value:
                color = "white" if value >= 0.65 * vmax else "#121211"
                ax_heat.text(xi, yi, str(value), ha="center", va="center", fontsize=7.0, color=color, fontweight="bold")

    ax_heat.set_xticks(x)
    ax_heat.set_xticklabels(bins)
    ax_heat.set_yticks(np.arange(len(groups)))
    ax_heat.set_yticklabels(groups)
    ax_heat.set_xlabel("Time bin (seconds)")
    ax_heat.set_ylabel("Failure group")
    ax_heat.set_xticks(np.arange(-0.5, len(bins), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.0)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.tick_params(axis="both", length=0)
    ax_heat.set_title("(c) Real-failure heatmap by failure group", loc="left", fontsize=8.2, pad=2)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    row_vmax = max(1, row_totals.max())
    total_image = ax_total.imshow(row_totals[:, None], aspect="auto", cmap=cmap, vmin=0, vmax=row_vmax, interpolation="nearest")
    for yi, value in enumerate(row_totals):
        ax_total.text(0, yi, str(value), ha="center", va="center", fontsize=7.0, color="white" if value >= 0.65 * row_vmax else "#121211", fontweight="bold")
    ax_total.set_xticks([0])
    ax_total.set_xticklabels(["sum"])
    ax_total.set_yticks(np.arange(len(groups)))
    ax_total.tick_params(axis="y", labelleft=False, left=False)
    ax_total.tick_params(axis="x", length=0)
    ax_total.set_xticks(np.arange(-0.5, 1.5, 1), minor=True)
    ax_total.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
    ax_total.grid(which="minor", color="white", linewidth=1.0)
    ax_total.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_total.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks([0, max(1, int(round(0.5 * vmax))), int(vmax)])
    cb.set_label("Verified spans", rotation=270, labelpad=9)
    cb.outline.set_linewidth(0.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "temporal_evidence_heatmap.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / "temporal_evidence_heatmap.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw_profile()
    draw_real_profile()
    draw_heatmap()
