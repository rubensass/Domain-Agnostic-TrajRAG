"""
Figure 3 replacement: forest plot of the mean paired difference
(baseline - TrajRAG, km) with 95% bootstrap CI (10,000 resamples of
voyages), Houston main set (n=50), one row per baseline x phase. Positive
= TrajRAG more accurate. Vertical dashed line at zero.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"

PHASE_ORDER = ["departure", "transit", "arrival"]
METHOD_ORDER = ["Kalman", "ARIMA", "Numeric-kNN", "AIS-ACNet", "LSTM", "BiLSTM", "Mean-Retrieval"]

if __name__ == "__main__":
    data = json.load(open(RES_DIR / "task2_task3_full_stats.json"))
    entries = {e["label"]: e for e in data["F1_houston_main"]}

    rows = []
    for method in METHOD_ORDER:
        for phase in PHASE_ORDER:
            label = f"Houston/{phase}/{method}"
            e = entries[label]
            rows.append((f"{method} ({phase[:4].capitalize()}.)", e["mean_diff_km"],
                         e["mean_diff_ci95"][0], e["mean_diff_ci95"][1], e["significant_holm_0.05"]))

    rows = rows[::-1]  # top-to-bottom = first method first, after a final flip for plotting order
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    lo = [r[1] - r[2] for r in rows]
    hi = [r[3] - r[1] for r in rows]
    colors = ["#1b9e77" if r[4] else "#999999" for r in rows]

    fig, ax = plt.subplots(figsize=(7, 8))
    y = list(range(len(rows)))
    for yi, m, l, h, col in zip(y, means, lo, hi, colors):
        ax.errorbar([m], [yi], xerr=[[l], [h]], fmt="o", color="black", ecolor=col,
                    elinewidth=2.5, capsize=3, markersize=4, markerfacecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean paired difference, baseline $-$ TrajRAG (km)\n"
                  "positive = TrajRAG more accurate")
    ax.set_title("Houston Ship Channel ($n=50$): mean difference per baseline and phase,\n"
                  "95% bootstrap CI (10,000 resamples of voyages), Holm-corrected within family",
                  fontsize=10)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#1b9e77", lw=2.5, label="Significant after Holm correction"),
        Line2D([0], [0], color="#999999", lw=2.5, label="Not significant after Holm correction"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "forest_plot_houston_main.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)
