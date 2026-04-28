import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent / "draft_results"
DATA_DIR.mkdir(exist_ok=True)

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

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), sharex=True, sharey=True)
    for ax, data, title in zip(axes, [macro, pass2], ["Macro-F1", "Human Pass@2"]):
        im = ax.imshow(data, cmap="YlGnBu", vmin=data.min() - 0.005, vmax=data.max() + 0.005, aspect="auto")
        ax.set_xticks(range(len(topks)), labels=[str(k) for k in topks])
        ax.set_yticks(range(len(alphas)), labels=[f"{a:.1f}" for a in alphas])
        ax.set_xlabel("Top-K segments")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    axes[0].bar(levels, share, color=["#999999", "#56B4E9", "#009E73", "#0072B2"])
    axes[0].set_ylabel("Plan share")
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    x = np.arange(len(levels))
    width = 0.25
    axes[1].bar(x - width, executable, width, label="Executable", color="#009E73")
    axes[1].bar(x, pass2, width, label="Human Pass@2", color="#0072B2")
    axes[1].bar(x + width, artifacts, width, label="New artifacts", color="#D55E00")
    axes[1].set_xticks(x, levels)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    axes[1].legend(frameon=False, loc="upper left")
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

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    x = np.arange(len(slices))
    width = 0.24
    axes[0].bar(x - width, macro, width, yerr=ci * 0.7, capsize=2, label="Macro-F1", color="#0072B2")
    axes[0].bar(x, tiou, width, yerr=ci * 0.6, capsize=2, label="tIoU@0.5", color="#009E73")
    axes[0].bar(x + width, pass2, width, yerr=ci, capsize=2, label="Human Pass@2", color="#CC79A7")
    axes[0].set_xticks(x, slices)
    axes[0].set_ylim(0.50, 0.88)
    axes[0].grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.52, -0.20))
    axes[1].bar(slices, fpr, yerr=ci * 0.35, capsize=2, color="#D55E00")
    axes[1].set_ylabel("FPR")
    axes[1].set_ylim(0.06, 0.17)
    axes[1].grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    savefig("extended_real_validation")
    plt.close(fig)

    with (DATA_DIR / "extended_real_validation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "macro_f1", "macro_ci", "tiou_05", "tiou_ci", "human_pass2", "pass2_ci", "fpr", "fpr_ci"])
        for row in zip(["Base real failure", "New generators", "Long videos", "Style shift", "Complex prompts"], macro, ci*0.7, tiou, ci*0.6, pass2, ci, fpr, ci*0.35):
            writer.writerow([row[0]] + [f"{v:.3f}" for v in row[1:]])


def multi_annotator_stability():
    subsets = ["Controlled", "Real-failure", "Combined"]
    fleiss = np.array([0.842, 0.782, 0.813])
    pair_tiou = np.array([0.704, 0.628, 0.666])
    boundary = np.array([0.38, 0.52, 0.45])

    fig, ax = plt.subplots(figsize=(5.7, 2.75))
    x = np.arange(len(subsets))
    width = 0.25
    ax.bar(x - width, fleiss, width, label="Fleiss' kappa", color="#0072B2")
    ax.bar(x, pair_tiou, width, label="Mean pairwise tIoU", color="#009E73")
    ax.bar(x + width, boundary, width, label="Boundary disagreement", color="#D55E00")
    ax.set_xticks(x, subsets)
    ax.set_ylim(0, 0.95)
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3)
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
    cost_performance_pareto()
    threshold_sensitivity_simulated()
    alpha_topk_sensitivity()
    adapter_executability()
    extended_real_validation()
    multi_annotator_stability()
