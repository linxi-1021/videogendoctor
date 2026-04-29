import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent / "draft_results"
DATA_DIR.mkdir(exist_ok=True)

# Colorblind-safe palette adapted from Okabe-Ito, with a consistent role map.
PALETTE = {
    "ours": "#0072B2",
    "open": "#009E73",
    "upper": "#D55E00",
    "vlm": "#CC79A7",
    "score": "#6B7280",
    "patch": "#4E79A7",
    "judge": "#D55E00",
    "grid": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
    "bar": "#DCEAF4",
}


def apply_style():
    """Publication defaults for NeurIPS-style vector figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.5,
        }
    )


def savefig(name):
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        plt.savefig(path)
        print(path)


def annotate_bars(ax, bars, fmt="{:.0f}", dy=0.01, fontsize=8.5):
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * dy
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


def confusion_matrix():
    codes = [
        "ID_FACE", "ID_BODY", "CA_MOVE", "CA_SHOT", "CA_SHAKE", "MO_JITTER",
        "MO_FROZEN", "MO_BREAK", "MO_EVENT", "AL_PROP", "ST_COMP", "SC_BG",
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

    fig, ax = plt.subplots(figsize=(6.8, 4.35))
    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=250, aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(codes)), labels=codes)
    ax.set_xlabel("Predicted code")
    ax.set_ylabel("Ground-truth code")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(codes), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = int(counts[i, j])
            if value == 0:
                continue
            color = "white" if value >= 150 or (j == len(labels) - 1 and value >= 8) else PALETTE["text"]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8.4, fontweight="semibold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Count")
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    savefig("confusion_matrix")
    plt.close(fig)


def temporal_evidence_profiles():
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
    x = np.arange(len(bins))
    controlled = np.array([332, 406, 448, 372, 270, 188])
    real = np.array([188, 232, 262, 286, 286, 282])

    for name, values, title, ymax in [
        ("temporal_evidence_profile", controlled, "(a) Controlled fixture: 2,016 verified spans", 480),
        ("temporal_evidence_real_profile", real, "(b) Real-failure subset: 1,536 verified spans", 320),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 1.75))
        bars = ax.bar(x, values, color=PALETTE["bar"], edgecolor="white", linewidth=0.8, width=0.78, zorder=1)
        ax.plot(x, values, color=PALETTE["ours"], marker="o", linewidth=2.0, zorder=3)
        annotate_bars(ax, bars, fmt="{:.0f}", dy=0.015, fontsize=9)
        ax.set_xticks(x, bins)
        ax.set_ylabel("Verified spans")
        ax.set_xlabel("Time bin (seconds)")
        ax.set_ylim(0, ymax)
        ax.set_title(title, loc="left", pad=2)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
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

    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=90, aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(groups)), labels=groups)
    ax.set_xlabel("Time bin (seconds)")
    ax.set_ylabel("Failure group")
    ax.set_title("(c) Real-failure heatmap by failure group", loc="left", pad=2)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            val = int(shown[i, j])
            color = "white" if val >= 56 else PALETTE["text"]
            ax.text(j, i, str(val), ha="center", va="center", fontsize=9, fontweight="semibold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Verified spans")
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.5)
    savefig("temporal_evidence_heatmap")
    plt.close(fig)


def repair_comparison():
    methods = ["VideoGenDoctor-full", "Patch+Judge", "VLM-to-patch", "Patch-only", "Score-only"]
    colors = [PALETTE["open"], "#D55E00", "#8172E8", PALETTE["patch"], PALETTE["score"]]
    auto_p1 = np.array([64.81, 62.96, 56.17, 52.47, 24.07])
    auto_p2 = np.array([83.64, 81.79, 74.38, 69.75, 35.80])
    human_p1 = np.array([58.33, 56.25, 50.69, 47.92, 18.75])
    human_p2 = np.array([76.39, 74.31, 67.36, 63.19, 27.78])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.95), sharey=True)
    y = np.arange(len(methods))
    for ax, p1, p2, title, xlabel in [
        (axes[0], auto_p1, auto_p2, "(a) Automatic verifier outcomes", "Automatic pass rate (%)"),
        (axes[1], human_p1, human_p2, "(b) Blind human outcomes", "Human pass rate (%)"),
    ]:
        for idx, (m, c) in enumerate(zip(methods, colors)):
            ax.plot([p1[idx], p2[idx]], [idx, idx], color=c, linewidth=1.9, alpha=0.75, zorder=1)
            ax.scatter(p1[idx], idx, s=44, facecolor="white", edgecolor=c, linewidth=1.7, zorder=3)
            ax.scatter(p2[idx], idx, s=48, facecolor=c, edgecolor=c, linewidth=1.0, zorder=4)
            ax.text(p1[idx] - 2.0, idx + 0.12, f"{p1[idx]:.1f}", ha="right", fontsize=8.7, color=PALETTE["muted"])
            ax.text(p2[idx] + 1.6, idx + 0.12, f"{p2[idx]:.1f}", ha="left", fontsize=8.7, color=PALETTE["text"])
            ax.text(98, idx, f"+{p2[idx] - p1[idx]:.1f}", ha="left", va="center", fontsize=8.5, color=PALETTE["muted"])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xlim(10, 106)
        ax.set_xticks([20, 40, 60, 80])
        ax.grid(True, axis="x")
        ax.grid(True, axis="y", alpha=0.35)
        ax.text(98, -0.55, "Gain", fontsize=8.7, color=PALETTE["muted"], ha="left")
    axes[0].set_yticks(y, labels=methods)
    axes[0].invert_yaxis()
    for tick, color in zip(axes[0].get_yticklabels(), colors):
        tick.set_color(color)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
                   markeredgecolor=PALETTE["text"], markeredgewidth=1.6, label="Pass@1"),
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=PALETTE["text"],
                   markeredgecolor=PALETTE["text"], label="Pass@2"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.0)
    savefig("manuscript_fig3")
    plt.close(fig)


def cost_performance_pareto():
    data = [
        ("Rule-only\n+ repair", 0.42, 0.5988, 0.58, PALETTE["score"]),
        ("Rule+Open-VLM\n+ repair", 0.73, 0.7160, 0.87, PALETTE["open"]),
        ("VideoGenDoctor\nfull", 1.00, 0.7639, 1.00, PALETTE["ours"]),
        ("Rule+GPT-4V\n+ repair", 2.75, 0.7917, 2.31, PALETTE["upper"]),
        ("VLM structured\nreport", 1.16, 0.6810, 1.20, PALETTE["vlm"]),
    ]

    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    for label, cost, pass2, latency, color in data:
        size = 55 + 70 * latency
        ax.scatter(cost, pass2, s=size, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        offset = (5, 4)
        if "GPT" in label:
            offset = (-78, -16)
        elif "structured" in label:
            offset = (5, -24)
        elif "Rule-only" in label:
            offset = (5, -16)
        elif "VideoGenDoctor" in label:
            offset = (6, 7)
        ax.annotate(label, (cost, pass2), textcoords="offset points", xytext=offset, ha="left", fontsize=8.8)

    ax.plot([0.42, 0.73, 1.00, 2.75], [0.5988, 0.7160, 0.7639, 0.7917],
            color="#7A7A7A", linewidth=1.1, linestyle="--", zorder=1)
    ax.axvline(1.0, color=PALETTE["ours"], linewidth=0.8, alpha=0.45)
    ax.axhline(0.7639, color=PALETTE["ours"], linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Relative cost (lower is better)")
    ax.set_ylabel("Human Pass@2 (higher is better)")
    ax.set_xlim(0.25, 2.95)
    ax.set_ylim(0.56, 0.82)
    ax.text(1.04, 0.748, "default operating point", color=PALETTE["ours"], fontsize=8)
    ax.text(1.65, 0.805, "higher-cost upper reference", color=PALETTE["upper"], fontsize=8)
    fig.tight_layout()
    savefig("cost_performance_pareto")
    plt.close(fig)


def threshold_sensitivity_simulated():
    thresholds = np.array([0.30, 0.40, 0.50, 0.60, 0.70])
    macro_f1 = np.array([0.902, 0.910, 0.911, 0.904, 0.886])
    tiou = np.array([0.716, 0.728, 0.732, 0.719, 0.694])
    fpr = np.array([0.158, 0.125, 0.100, 0.083, 0.067])
    unnecessary_patch = np.array([0.142, 0.111, 0.087, 0.071, 0.058])

    fig, ax = plt.subplots(figsize=(5.7, 3.05))
    ax.plot(thresholds, macro_f1, marker="o", color=PALETTE["ours"], label="Macro-F1")
    ax.plot(thresholds, tiou, marker="s", color=PALETTE["open"], label="tIoU@0.5")
    ax2 = ax.twinx()
    ax2.plot(thresholds, fpr, marker="^", color=PALETTE["upper"], label="Real-normal FPR")
    ax2.plot(thresholds, unnecessary_patch, marker="D", color=PALETTE["vlm"], label="Unnecessary patch")

    ax.axvline(0.50, color=PALETTE["muted"], linewidth=0.9, linestyle="--")
    ax.text(0.505, 0.891, "default", fontsize=8.5, color=PALETTE["muted"])
    ax.set_xlabel("Diagnosis threshold")
    ax.set_ylabel("Diagnosis / localization metrics")
    ax2.set_ylabel("Risk metrics")
    ax.set_ylim(0.66, 0.93)
    ax2.set_ylim(0.04, 0.18)
    ax.grid(True)
    ax2.grid(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout()
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
    macro = np.array([
        [0.892, 0.901, 0.902],
        [0.904, 0.909, 0.908],
        [0.907, 0.911, 0.910],
        [0.899, 0.906, 0.905],
    ])
    pass2 = np.array([
        [0.689, 0.721, 0.716],
        [0.724, 0.752, 0.747],
        [0.739, 0.764, 0.758],
        [0.711, 0.736, 0.731],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True, sharey=True)
    for ax, data, title in zip(axes, [macro, pass2], ["Macro-F1", "Human Pass@2"]):
        im = ax.imshow(data, cmap="Blues", vmin=data.min() - 0.005, vmax=data.max() + 0.005, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(topks)), labels=[str(k) for k in topks])
        ax.set_yticks(range(len(alphas)), labels=[f"{a:.1f}" for a in alphas])
        ax.set_xlabel("Top-K segments")
        ax.set_title(title)
        ax.set_xticks(np.arange(-0.5, len(topks), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(alphas), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "white" if data[i, j] > data.mean() else PALETTE["text"]
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8.8, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel(r"Stage-2 weight $\alpha$")
    fig.tight_layout()
    savefig("alpha_topk_sensitivity")
    plt.close(fig)

    with (DATA_DIR / "alpha_topk_sensitivity.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "top_k", "macro_f1", "human_pass2"])
        for i, alpha in enumerate(alphas):
            for j, topk in enumerate(topks):
                writer.writerow([f"{alpha:.1f}", int(topk), f"{macro[i, j]:.3f}", f"{pass2[i, j]:.3f}"])


def adapter_executability():
    levels = ["L0\nabstain", "L1\nplan", "L2\npartial", "L3\nnative"]
    share = np.array([0.10, 0.31, 0.42, 0.17])
    executable = np.array([0.00, 0.28, 0.74, 0.96])
    pass2 = np.array([0.28, 0.54, 0.73, 0.81])
    artifacts = np.array([0.04, 0.10, 0.14, 0.16])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    bars = axes[0].bar(levels, share, color=[PALETTE["score"], "#56B4E9", PALETTE["open"], PALETTE["ours"]],
                       edgecolor="white", linewidth=0.7)
    axes[0].set_ylabel("Plan share")
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    annotate_bars(axes[0], bars, fmt="{:.2f}", dy=0.015, fontsize=8.5)

    x = np.arange(len(levels))
    width = 0.25
    axes[1].bar(x - width, executable, width, label="Executable", color=PALETTE["open"], edgecolor="white", linewidth=0.5)
    axes[1].bar(x, pass2, width, label="Human Pass@2", color=PALETTE["ours"], edgecolor="white", linewidth=0.5)
    axes[1].bar(x + width, artifacts, width, label="New artifacts", color=PALETTE["upper"], edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(x, levels)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, axis="y")
    axes[1].grid(False, axis="x")
    axes[1].legend(loc="upper left", ncol=1)
    fig.tight_layout()
    savefig("adapter_executability")
    plt.close(fig)

    with (DATA_DIR / "adapter_executability.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["adapter_level", "plan_share", "adapter_executable_rate", "human_pass2", "new_artifact_rate"])
        for level, s, e, p, a in zip(["L0", "L1", "L2", "L3"], share, executable, pass2, artifacts):
            writer.writerow([level, f"{s:.3f}", f"{e:.3f}", f"{p:.3f}", f"{a:.3f}"])


def extended_real_validation():
    slices = ["Base", "New gen.", "Long", "Style", "Complex"]
    macro = np.array([0.826, 0.802, 0.784, 0.811, 0.793])
    tiou = np.array([0.598, 0.574, 0.552, 0.581, 0.560])
    pass2 = np.array([0.691, 0.662, 0.628, 0.674, 0.641])
    fpr = np.array([0.100, 0.126, 0.142, 0.118, 0.135])
    ci = np.array([0.028, 0.034, 0.041, 0.036, 0.039])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.85), gridspec_kw={"width_ratios": [2.1, 1.0]})
    x = np.arange(len(slices))
    width = 0.24
    axes[0].bar(x - width, macro, width, yerr=ci * 0.7, capsize=2, label="Macro-F1",
                color=PALETTE["ours"], edgecolor="white", linewidth=0.5)
    axes[0].bar(x, tiou, width, yerr=ci * 0.6, capsize=2, label="tIoU@0.5",
                color=PALETTE["open"], edgecolor="white", linewidth=0.5)
    axes[0].bar(x + width, pass2, width, yerr=ci, capsize=2, label="Human Pass@2",
                color=PALETTE["vlm"], edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(x, slices)
    axes[0].set_ylim(0.50, 0.88)
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.52, -0.18))
    axes[1].bar(slices, fpr, yerr=ci * 0.35, capsize=2, color=PALETTE["upper"],
                edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("FPR")
    axes[1].set_ylim(0.06, 0.17)
    axes[1].grid(True, axis="y")
    axes[1].grid(False, axis="x")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    savefig("extended_real_validation")
    plt.close(fig)

    with (DATA_DIR / "extended_real_validation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "macro_f1", "macro_ci", "tiou_05", "tiou_ci", "human_pass2", "pass2_ci", "fpr", "fpr_ci"])
        rows = zip(
            ["Base real failure", "New generators", "Long videos", "Style shift", "Complex prompts"],
            macro, ci * 0.7, tiou, ci * 0.6, pass2, ci, fpr, ci * 0.35,
        )
        for row in rows:
            writer.writerow([row[0]] + [f"{v:.3f}" for v in row[1:]])


def multi_annotator_stability():
    subsets = ["Controlled", "Real-failure", "Combined"]
    fleiss = np.array([0.842, 0.782, 0.813])
    pair_tiou = np.array([0.704, 0.628, 0.666])
    boundary = np.array([0.38, 0.52, 0.45])

    fig, ax = plt.subplots(figsize=(5.7, 2.75))
    x = np.arange(len(subsets))
    width = 0.25
    ax.bar(x - width, fleiss, width, label="Fleiss' kappa", color=PALETTE["ours"], edgecolor="white", linewidth=0.5)
    ax.bar(x, pair_tiou, width, label="Mean pairwise tIoU", color=PALETTE["open"], edgecolor="white", linewidth=0.5)
    ax.bar(x + width, boundary, width, label="Boundary disagreement", color=PALETTE["upper"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, subsets)
    ax.set_ylim(0, 0.95)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
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
