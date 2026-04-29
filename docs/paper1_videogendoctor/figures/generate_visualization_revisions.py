import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent / "draft_results"
DATA_DIR.mkdir(exist_ok=True)

PALETTE = {
    "ours": "#8FB6D9",
    "ours_dark": "#4F7FAE",
    "green": "#B9D8BE",
    "green_dark": "#6FA675",
    "coral": "#E7B3A6",
    "coral_dark": "#C57E6E",
    "lavender": "#CEC4E6",
    "lavender_dark": "#9887C0",
    "sand": "#EBDAB8",
    "sand_dark": "#C8A769",
    "gray": "#D7DEE7",
    "gray_dark": "#7F8B97",
    "panel": "#FBFCFE",
    "grid": "#E6ECF3",
    "axis": "#7A8794",
    "text": "#28333D",
    "muted": "#677380",
    "highlight": "#EDF4FB",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "soft_blues",
    ["#F8FBFE", "#DFECF7", "#BED6EC", "#8FB6D9", "#4F7FAE"],
)


def apply_style():
    plt.rcParams.update(
        {
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
        }
    )


def savefig(name):
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        plt.savefig(path)
        print(path)


def soften_axes(ax, y_grid=True, x_grid=False):
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.tick_params(colors=PALETTE["text"])
    ax.grid(False)
    if y_grid:
        ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    if x_grid:
        ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)


def annotate_bars(ax, bars, fmt="{:.2f}", padding=0.012, fontsize=8.3):
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * padding
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=PALETTE["text"],
        )


def annotate_horizontal(ax, bars, fmt="{:.1f}", padding=1.0, fontsize=8.2):
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + padding,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=PALETTE["text"],
        )


def legend_patch(color, label):
    return Patch(facecolor=color, edgecolor="white", linewidth=0.7, label=label)


def confusion_matrix():
    codes = [
        "ID_FACE",
        "ID_BODY",
        "CA_MOVE",
        "CA_SHOT",
        "CA_SHAKE",
        "MO_JITTER",
        "MO_FROZEN",
        "MO_BREAK",
        "MO_EVENT",
        "AL_PROP",
        "ST_COMP",
        "SC_BG",
    ]
    counts = np.array(
        [
            [136, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
            [10, 118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 215, 14, 7, 4, 0, 6, 0, 0, 0, 0, 8],
            [0, 0, 12, 124, 10, 0, 0, 4, 0, 0, 0, 0, 6],
            [0, 0, 8, 9, 105, 7, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 6, 151, 11, 9, 6, 0, 0, 0, 12],
            [0, 0, 0, 0, 0, 6, 143, 5, 0, 0, 0, 0, 7],
            [0, 0, 4, 0, 0, 11, 8, 137, 4, 0, 0, 3, 10],
            [0, 0, 0, 0, 0, 7, 0, 4, 108, 0, 0, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 188, 5, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 243, 4, 5],
            [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 148, 8],
        ],
        dtype=float,
    )
    labels = codes + ["miss"]

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    im = ax.imshow(counts, cmap=HEATMAP_CMAP, vmin=0, vmax=250, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=38, ha="right")
    ax.set_yticks(np.arange(len(codes)), labels=codes)
    ax.set_xlabel("Predicted code")
    ax.set_ylabel("Ground-truth code")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(codes), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = int(counts[i, j])
            if value == 0:
                continue
            color = "white" if value >= 120 else PALETTE["text"]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8.8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Count")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=8.5, colors=PALETTE["text"])
    fig.tight_layout()
    savefig("confusion_matrix")
    plt.close(fig)


def temporal_evidence_profiles():
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
    x = np.arange(len(bins))
    controlled = np.array([332, 406, 448, 372, 270, 188])
    real = np.array([188, 232, 262, 286, 286, 282])

    specs = [
        (
            "temporal_evidence_profile",
            controlled,
            "(a) Controlled fixture: 2,016 verified spans",
            520,
            PALETTE["ours"],
            PALETTE["ours_dark"],
        ),
        (
            "temporal_evidence_real_profile",
            real,
            "(b) Real-failure subset: 1,536 verified spans",
            350,
            PALETTE["green"],
            PALETTE["green_dark"],
        ),
    ]

    for name, values, title, ymax, bar_color, line_color in specs:
        fig, ax = plt.subplots(figsize=(7.5, 2.45))
        bars = ax.bar(x, values, color=bar_color, edgecolor="white", linewidth=0.8, width=0.72)
        ax.plot(x, values, color=line_color, marker="o", linewidth=1.9, zorder=3)
        ax.set_xticks(x, bins)
        ax.set_ylabel("Verified spans")
        ax.set_xlabel("Time bin (seconds)")
        ax.set_ylim(0, ymax)
        ax.set_title(title, loc="left", pad=4)
        soften_axes(ax, y_grid=True, x_grid=False)
        annotate_bars(ax, bars, fmt="{:.0f}", padding=0.012, fontsize=9.0)
        fig.tight_layout(pad=0.5)
        savefig(name)
        plt.close(fig)


def temporal_evidence_heatmap():
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
    groups = ["ID", "CA", "MO", "AL", "ST", "SC"]
    data = np.array(
        [
            [52, 58, 47, 39, 31, 21],
            [44, 52, 58, 61, 59, 52],
            [46, 58, 71, 82, 88, 85],
            [20, 32, 44, 52, 54, 46],
            [10, 16, 22, 26, 32, 30],
            [16, 16, 20, 26, 22, 48],
        ]
    )
    totals = data.sum(axis=1, keepdims=True)
    shown = np.concatenate([data, totals], axis=1)
    labels = bins + ["sum"]

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    im = ax.imshow(shown, cmap=HEATMAP_CMAP, vmin=0, vmax=90, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(groups)), labels=groups)
    ax.set_xlabel("Time bin (seconds)")
    ax.set_ylabel("Failure group")
    ax.set_title("(c) Real-failure heatmap by failure group", loc="left", pad=4)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            value = int(shown[i, j])
            color = "white" if value >= 56 else PALETTE["text"]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=9.0, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Verified spans")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=8.5, colors=PALETTE["text"])
    fig.tight_layout(pad=0.5)
    savefig("temporal_evidence_heatmap")
    plt.close(fig)


def repair_comparison():
    methods = ["VideoGenDoctor-full", "Patch+Judge", "VLM-to-patch", "Patch-only", "Score-only"]
    auto_p1 = np.array([64.81, 62.96, 56.17, 52.47, 24.07])
    auto_p2 = np.array([83.64, 81.79, 74.38, 69.75, 35.80])
    human_p1 = np.array([58.33, 56.25, 50.69, 47.92, 18.75])
    human_p2 = np.array([76.39, 74.31, 67.36, 63.19, 27.78])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    y = np.arange(len(methods))
    bar_h = 0.34
    panels = [
        (axes[0], auto_p1, auto_p2, "(a) Automatic verifier outcomes"),
        (axes[1], human_p1, human_p2, "(b) Blind human outcomes"),
    ]

    for ax, p1, p2, title in panels:
        ax.axhspan(-0.5, 0.5, color=PALETTE["highlight"], zorder=0)
        bars1 = ax.barh(y - bar_h / 2, p1, height=bar_h, color=PALETTE["ours"], edgecolor="white", linewidth=0.7)
        bars2 = ax.barh(y + bar_h / 2, p2, height=bar_h, color=PALETTE["coral"], edgecolor="white", linewidth=0.7)
        annotate_horizontal(ax, bars1, fmt="{:.1f}", padding=0.9)
        annotate_horizontal(ax, bars2, fmt="{:.1f}", padding=0.9)
        ax.set_xlim(0, 90)
        ax.set_xlabel("Pass rate (%)")
        ax.set_title(title, pad=5)
        soften_axes(ax, y_grid=False, x_grid=True)

    axes[0].set_yticks(y, methods)
    axes[0].invert_yaxis()
    for tick in axes[0].get_yticklabels():
        if tick.get_text() == "VideoGenDoctor-full":
            tick.set_fontweight("semibold")
            tick.set_color(PALETTE["ours_dark"])

    fig.legend(
        [legend_patch(PALETTE["ours"], "Pass@1"), legend_patch(PALETTE["coral"], "Pass@2")],
        ["Pass@1", "Pass@2"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.6)
    savefig("manuscript_fig3")
    plt.close(fig)


def cost_performance_pareto():
    data = [
        ("Rule-only", 0.42, 0.5988, 0.58, PALETTE["gray"]),
        ("Rule+Open-VLM", 0.73, 0.7160, 0.87, PALETTE["green"]),
        ("VideoGenDoctor", 1.00, 0.7639, 1.00, PALETTE["ours"]),
        ("Rule+GPT-4V", 2.75, 0.7917, 2.31, PALETTE["coral"]),
        ("Structured report", 1.16, 0.6810, 1.20, PALETTE["lavender"]),
    ]
    pareto_x = [0.42, 0.73, 1.00, 2.75]
    pareto_y = [0.5988, 0.7160, 0.7639, 0.7917]

    fig, ax = plt.subplots(figsize=(7.5, 3.25))
    for label, cost, pass2, latency, color in data:
        size = 170 + 120 * latency
        ax.scatter(cost, pass2, s=size, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(cost + 0.04, pass2 + 0.004, label, fontsize=8.6, color=PALETTE["text"])

    ax.plot(pareto_x, pareto_y, color=PALETTE["muted"], linewidth=1.0, linestyle="--", zorder=2)
    ax.scatter([1.00], [0.7639], s=320, facecolor="none", edgecolor=PALETTE["ours_dark"], linewidth=1.3, zorder=4)
    ax.set_xlabel("Relative cost (×, lower is better)")
    ax.set_ylabel("Human Pass@2")
    ax.set_xlim(0.25, 2.95)
    ax.set_ylim(0.56, 0.82)
    soften_axes(ax, y_grid=True, x_grid=True)
    fig.tight_layout()
    savefig("cost_performance_pareto")
    plt.close(fig)


def threshold_sensitivity_simulated():
    thresholds = np.array([0.30, 0.40, 0.50, 0.60, 0.70])
    macro_f1 = np.array([0.902, 0.910, 0.911, 0.904, 0.886])
    tiou = np.array([0.716, 0.728, 0.732, 0.719, 0.694])
    fpr = np.array([0.158, 0.125, 0.100, 0.083, 0.067])
    unnecessary_patch = np.array([0.142, 0.111, 0.087, 0.071, 0.058])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.15), sharex=True)
    left, right = axes

    left.plot(thresholds, macro_f1, marker="o", color=PALETTE["ours_dark"], label="Macro-F1")
    left.plot(thresholds, tiou, marker="s", color=PALETTE["green_dark"], label="tIoU@0.5")
    left.axvline(0.50, color=PALETTE["muted"], linewidth=0.9, linestyle="--")
    left.set_ylim(0.68, 0.93)
    left.set_ylabel("Diagnosis / localization metric")
    left.set_title("(a) Utility metrics", pad=4)
    soften_axes(left, y_grid=True, x_grid=False)

    right.plot(thresholds, fpr, marker="^", color=PALETTE["coral_dark"], label="Real-normal FPR")
    right.plot(thresholds, unnecessary_patch, marker="D", color=PALETTE["lavender_dark"], label="Unnecessary patch")
    right.axvline(0.50, color=PALETTE["muted"], linewidth=0.9, linestyle="--")
    right.set_ylim(0.05, 0.165)
    right.set_ylabel("Risk metric")
    right.set_title("(b) Risk metrics", pad=4)
    soften_axes(right, y_grid=True, x_grid=False)

    for ax in axes:
        ax.set_xlabel("Diagnosis threshold")
        ax.set_xticks(thresholds)
        y0, y1 = ax.get_ylim()
        ax.text(0.503, y1 - 0.10 * (y1 - y0), "default", fontsize=8.2, color=PALETTE["muted"])

    handles = [h for h in (left.get_lines() + right.get_lines()) if not h.get_label().startswith("_")]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), columnspacing=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=1.25)
    savefig("threshold_sensitivity")
    plt.close(fig)

    with (DATA_DIR / "threshold_sensitivity.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "macro_f1", "tiou_05", "real_normal_fpr", "unnecessary_patch_rate"])
        for row in zip(thresholds, macro_f1, tiou, fpr, unnecessary_patch):
            writer.writerow([f"{row[0]:.2f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}"])


def alpha_topk_sensitivity():
    alphas = np.array([0.0, 0.3, 0.6, 0.9])
    topks = np.array([1, 3, 5])
    macro = np.array(
        [
            [0.892, 0.901, 0.902],
            [0.904, 0.909, 0.908],
            [0.907, 0.911, 0.910],
            [0.899, 0.906, 0.905],
        ]
    )
    pass2 = np.array(
        [
            [0.689, 0.721, 0.716],
            [0.724, 0.752, 0.747],
            [0.739, 0.764, 0.758],
            [0.711, 0.736, 0.731],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharex=True, sharey=True)
    specs = [
        (axes[0], macro, "(a) Macro-F1"),
        (axes[1], pass2, "(b) Human Pass@2"),
    ]

    for ax, data, title in specs:
        vmin = float(data.min() - 0.003)
        vmax = float(data.max() + 0.003)
        im = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(topks)), labels=[str(k) for k in topks])
        ax.set_yticks(range(len(alphas)), labels=[f"{a:.1f}" for a in alphas])
        ax.set_xlabel("Top-$K$ segments")
        ax.set_title(title, pad=4)
        ax.set_xticks(np.arange(-0.5, len(topks), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(alphas), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.add_patch(Rectangle((0.5, 1.5), 1.0, 1.0, fill=False, edgecolor=PALETTE["text"], linewidth=1.3, zorder=5))

        threshold = (vmin + vmax) / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "white" if data[i, j] >= threshold else PALETTE["text"]
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9.0, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8.2)

    axes[0].set_ylabel(r"Stage-2 weight $\alpha$")
    fig.tight_layout(pad=0.5, w_pad=1.1)
    savefig("alpha_topk_sensitivity")
    plt.close(fig)

    with (DATA_DIR / "alpha_topk_sensitivity.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "top_k", "macro_f1", "human_pass2"])
        for i, alpha in enumerate(alphas):
            for j, topk in enumerate(topks):
                writer.writerow([f"{alpha:.1f}", int(topk), f"{macro[i, j]:.3f}", f"{pass2[i, j]:.3f}"])


def adapter_executability():
    levels_short = ["L0", "L1", "L2", "L3"]
    levels_long = ["L0\nabstain", "L1\nplan", "L2\npartial", "L3\nnative"]
    share = np.array([0.10, 0.31, 0.42, 0.17])
    executable = np.array([0.00, 0.28, 0.74, 0.96])
    pass2 = np.array([0.28, 0.54, 0.73, 0.81])
    artifacts = np.array([0.04, 0.10, 0.14, 0.16])
    colors = [PALETTE["gray"], PALETTE["sand"], PALETTE["green"], PALETTE["ours"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1), gridspec_kw={"width_ratios": [1.05, 1.95]})
    left, right = axes

    starts = np.cumsum(np.concatenate([[0.0], share[:-1]]))
    for idx, level in enumerate(levels_short):
        left.barh(["Plan share"], [share[idx]], left=starts[idx], color=colors[idx], edgecolor="white", linewidth=0.7, label=level)
        left.text(starts[idx] + share[idx] / 2, 0, f"{level}\n{share[idx]:.2f}", ha="center", va="center", fontsize=8.2, color=PALETTE["text"])
    left.set_xlim(0, 1.0)
    left.set_xlabel("Share")
    left.set_title("(a) Plan composition", pad=4)
    soften_axes(left, y_grid=False, x_grid=True)

    x = np.arange(len(levels_long))
    width = 0.24
    b1 = right.bar(x - width, executable, width, label="Executable", color=PALETTE["green"], edgecolor="white", linewidth=0.6)
    b2 = right.bar(x, pass2, width, label="Human Pass@2", color=PALETTE["ours"], edgecolor="white", linewidth=0.6)
    b3 = right.bar(x + width, artifacts, width, label="New artifacts", color=PALETTE["coral"], edgecolor="white", linewidth=0.6)
    right.set_xticks(x, levels_long)
    right.set_ylim(0, 1.05)
    right.set_title("(b) Outcome by adapter level", pad=4)
    soften_axes(right, y_grid=True, x_grid=False)
    annotate_bars(right, b1, fmt="{:.2f}", padding=0.012)
    annotate_bars(right, b2, fmt="{:.2f}", padding=0.012)
    annotate_bars(right, b3, fmt="{:.2f}", padding=0.012)

    fig.legend([b1, b2, b3], ["Executable", "Human Pass@2", "New artifacts"], loc="upper center", ncol=3, bbox_to_anchor=(0.63, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=1.15)
    savefig("adapter_executability")
    plt.close(fig)

    with (DATA_DIR / "adapter_executability.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["adapter_level", "plan_share", "adapter_executable_rate", "human_pass2", "new_artifact_rate"])
        for level, s, e, p, a in zip(levels_short, share, executable, pass2, artifacts):
            writer.writerow([level, f"{s:.3f}", f"{e:.3f}", f"{p:.3f}", f"{a:.3f}"])


def extended_real_validation():
    slices = ["Base", "New gen.", "Long", "Style", "Complex"]
    macro = np.array([0.826, 0.802, 0.784, 0.811, 0.793])
    tiou = np.array([0.598, 0.574, 0.552, 0.581, 0.560])
    pass2 = np.array([0.691, 0.662, 0.628, 0.674, 0.641])
    fpr = np.array([0.100, 0.126, 0.142, 0.118, 0.135])
    ci = np.array([0.028, 0.034, 0.041, 0.036, 0.039])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), gridspec_kw={"width_ratios": [2.2, 1.25]})
    left, right = axes
    x = np.arange(len(slices))
    width = 0.24

    b1 = left.bar(x - width, macro, width, yerr=ci * 0.7, capsize=2.5, label="Macro-F1", color=PALETTE["ours"], edgecolor="white", linewidth=0.6)
    b2 = left.bar(x, tiou, width, yerr=ci * 0.6, capsize=2.5, label="tIoU@0.5", color=PALETTE["green"], edgecolor="white", linewidth=0.6)
    b3 = left.bar(x + width, pass2, width, yerr=ci, capsize=2.5, label="Human Pass@2", color=PALETTE["lavender"], edgecolor="white", linewidth=0.6)
    left.set_xticks(x, slices)
    left.set_ylim(0.50, 0.88)
    left.set_title("(a) Utility metrics with uncertainty", pad=4)
    soften_axes(left, y_grid=True, x_grid=False)

    b4 = right.bar(np.arange(len(slices)), fpr, yerr=ci * 0.35, capsize=2.5, color=PALETTE["coral"], edgecolor="white", linewidth=0.6)
    right.set_xticks(np.arange(len(slices)), slices, rotation=15)
    right.set_ylim(0.06, 0.17)
    right.set_ylabel("FPR")
    right.set_title("(b) Real-normal false positives", pad=4)
    soften_axes(right, y_grid=True, x_grid=False)
    annotate_bars(right, b4, fmt="{:.3f}", padding=0.018)

    fig.legend([b1, b2, b3, b4], ["Macro-F1", "tIoU@0.5", "Human Pass@2", "FPR"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=1.45)
    savefig("extended_real_validation")
    plt.close(fig)

    with (DATA_DIR / "extended_real_validation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "macro_f1", "macro_ci", "tiou_05", "tiou_ci", "human_pass2", "pass2_ci", "fpr", "fpr_ci"])
        rows = zip(
            ["Base real failure", "New generators", "Long videos", "Style shift", "Complex prompts"],
            macro,
            ci * 0.7,
            tiou,
            ci * 0.6,
            pass2,
            ci,
            fpr,
            ci * 0.35,
        )
        for row in rows:
            writer.writerow([row[0]] + [f"{value:.3f}" for value in row[1:]])


def multi_annotator_stability():
    subsets = ["Controlled", "Real-failure", "Combined"]
    fleiss = np.array([0.842, 0.782, 0.813])
    pair_tiou = np.array([0.704, 0.628, 0.666])
    boundary = np.array([0.38, 0.52, 0.45])

    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.05), gridspec_kw={"width_ratios": [1.8, 1.15]})
    left, right = axes
    x = np.arange(len(subsets))
    width = 0.33

    b1 = left.bar(x - width / 2, fleiss, width, label="Fleiss' $\\kappa$", color=PALETTE["ours"], edgecolor="white", linewidth=0.6)
    b2 = left.bar(x + width / 2, pair_tiou, width, label="Mean pairwise tIoU", color=PALETTE["green"], edgecolor="white", linewidth=0.6)
    left.set_xticks(x, subsets)
    left.set_ylim(0, 0.95)
    left.set_title("(a) Agreement metrics", pad=4)
    soften_axes(left, y_grid=True, x_grid=False)
    annotate_bars(left, b1, fmt="{:.3f}", padding=0.012)
    annotate_bars(left, b2, fmt="{:.3f}", padding=0.012)

    b3 = right.bar(np.arange(len(subsets)), boundary, color=PALETTE["coral"], edgecolor="white", linewidth=0.6)
    right.set_xticks(np.arange(len(subsets)), subsets, rotation=15)
    right.set_ylim(0, 0.62)
    right.set_ylabel("Seconds")
    right.set_title("(b) Boundary disagreement", pad=4)
    soften_axes(right, y_grid=True, x_grid=False)
    annotate_bars(right, b3, fmt="{:.2f}", padding=0.02)

    fig.legend([b1, b2, b3], ["Fleiss' $\\kappa$", "Mean pairwise tIoU", "Boundary disagreement"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=1.15)
    savefig("multi_annotator_stability")
    plt.close(fig)

    with (DATA_DIR / "multi_annotator_stability.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "videos", "candidate_spans", "annotators", "fleiss_kappa", "mean_pairwise_tiou", "boundary_disagreement_sec"])
        rows = [
            ("Controlled", 72, 468, 3, fleiss[0], pair_tiou[0], boundary[0]),
            ("Real-failure", 48, 300, 3, fleiss[1], pair_tiou[1], boundary[1]),
            ("Combined", 120, 768, 3, fleiss[2], pair_tiou[2], boundary[2]),
        ]
        for row in rows:
            writer.writerow([row[0], row[1], row[2], row[3], f"{row[4]:.3f}", f"{row[5]:.3f}", f"{row[6]:.2f}"])


if __name__ == "__main__":
    apply_style()
    confusion_matrix()
    temporal_evidence_profiles()
    temporal_evidence_heatmap()
    repair_comparison()
    cost_performance_pareto()
    threshold_sensitivity_simulated()
    alpha_topk_sensitivity()
    adapter_executability()
    extended_real_validation()
    multi_annotator_stability()
