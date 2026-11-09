"""
Task 2: paired bootstrap CIs of the MEAN difference (baseline - TrajRAG),
resampling voyage IDs (not points), 10,000 resamples, per port/phase/
experiment. Produces the data for the Figure-3 replacement.

Task 3: completes the paired Wilcoxon tests that were not yet in
final_significance_all.json (LA was actually already there -- verified
against sec_results.tex's own tab:corridor numbers; what's genuinely new is
the short-horizon TrajRAG-vs-AIS-ACNet comparison) and applies Holm's
step-down correction within four pre-registered families:
  F1: Houston main, TrajRAG vs {Kalman,ARIMA,LSTM,BiLSTM,Mean-Retrieval,
      Numeric-kNN,AIS-ACNet} x 3 phases (21 tests)
  F2: LA/Long Beach main, TrajRAG vs {Kalman,ARIMA,LSTM,BiLSTM,
      Mean-Retrieval,AIS-ACNet} x 3 phases (18 tests; Numeric-kNN was never
      rescored for LA -- reported as a limitation, not invented)
  F3: Short-horizon matched, TrajRAG vs AIS-ACNet x 3 phases (3 tests;
      iTransformer and the other modern forecasters excluded -- no
      per-voyage data was ever saved for them, only aggregate ADE/FDE, so no
      paired test is possible against them from saved outputs)
  F4: Maritime ablations, Houston, TrajRAG vs {no_retrieval,no_guard,
      no_heading} x 3 phases (9 tests)

scipy's wilcoxon is recomputed independently (not just copied from
final_significance_all.json) for every cell, including the ones already
present, as a consistency check.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import pairing as P
import common as c

N_BOOT = 10000
RNG_SEED = 42


def bootstrap_mean_diff_ci(diff, n_boot=N_BOOT, seed=RNG_SEED):
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(trajrag_dict, baseline_dict, label):
    ids, t, b = P.matched_pairs(trajrag_dict, baseline_dict)
    if len(ids) < 2:
        return {"label": label, "n": len(ids), "error": "fewer than 2 matched voyages"}
    t = np.array(t, dtype=float)
    b = np.array(b, dtype=float)
    diff = b - t  # positive => TrajRAG better
    try:
        stat, p = wilcoxon(b, t)
    except ValueError as e:
        p = None
    ci_lo, ci_hi = bootstrap_mean_diff_ci(diff)
    return {
        "label": label,
        "n": len(ids),
        "trajrag_mean_km": round(float(t.mean()), 4),
        "baseline_mean_km": round(float(b.mean()), 4),
        "mean_diff_km": round(float(diff.mean()), 4),
        "mean_diff_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "median_diff_km": round(float(np.median(diff)), 4),
        "wilcoxon_p": p,
    }


def holm_correction(entries, p_key="wilcoxon_p"):
    """entries: list of dicts with p_key; adds 'p_holm' and 'significant_holm'
    (alpha=0.05) in place, Holm-Bonferroni step-down."""
    valid = [(i, e[p_key]) for i, e in enumerate(entries) if e.get(p_key) is not None]
    valid.sort(key=lambda x: x[1])
    m = len(valid)
    running_max = 0.0
    for rank, (i, p) in enumerate(valid):
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        adj = min(running_max, 1.0)
        entries[i]["p_holm"] = round(float(adj), 6)
        entries[i]["significant_holm_0.05"] = bool(adj < 0.05)
    for e in entries:
        if p_key not in e or e.get(p_key) is None:
            e.setdefault("p_holm", None)
            e.setdefault("significant_holm_0.05", None)
    return entries


def family_houston_main():
    methods_fn = {
        "Kalman": lambda ph: P.classical_baseline_errors("HOU-SHIPCHANNEL", "Kalman", ph),
        "ARIMA": lambda ph: P.classical_baseline_errors("HOU-SHIPCHANNEL", "ARIMA", ph),
        "LSTM": lambda ph: P.classical_baseline_errors("HOU-SHIPCHANNEL", "LSTM", ph),
        "BiLSTM": lambda ph: P.classical_baseline_errors("HOU-SHIPCHANNEL", "BiLSTM", ph),
        "Numeric-kNN": lambda ph: P.classical_baseline_errors("HOU-SHIPCHANNEL", "Numeric-kNN", ph),
        "Mean-Retrieval": lambda ph: P.mean_retrieval_main_errors("HOU-SHIPCHANNEL", ph),
        "AIS-ACNet": lambda ph: P.aisacnet_main_errors("HOU-SHIPCHANNEL", ph),
    }
    out = []
    for phase in c.PHASES:
        traj = P.trajrag_main_errors("HOU-SHIPCHANNEL", phase)
        for method, fn in methods_fn.items():
            base = fn(phase)
            out.append(paired_stats(traj, base, f"Houston/{phase}/{method}"))
    return out


def family_la_main():
    methods_fn = {
        "Kalman": lambda ph: P.classical_baseline_errors("LA-LONGBEACH", "Kalman", ph),
        "ARIMA": lambda ph: P.classical_baseline_errors("LA-LONGBEACH", "ARIMA", ph),
        "LSTM": lambda ph: P.classical_baseline_errors("LA-LONGBEACH", "LSTM", ph),
        "BiLSTM": lambda ph: P.classical_baseline_errors("LA-LONGBEACH", "BiLSTM", ph),
        "Mean-Retrieval": lambda ph: P.mean_retrieval_main_errors("LA-LONGBEACH", ph),
        "AIS-ACNet": lambda ph: P.aisacnet_main_errors("LA-LONGBEACH", ph),
    }
    out = []
    for phase in c.PHASES:
        traj = P.trajrag_main_errors("LA-LONGBEACH", phase)
        for method, fn in methods_fn.items():
            base = fn(phase)
            out.append(paired_stats(traj, base, f"LA-LongBeach/{phase}/{method}"))
    out.append({
        "label": "LA-LongBeach/ALL/Numeric-kNN",
        "n": 0,
        "error": "Numeric-kNN was never rescored on LA/Long Beach (baseline_rescore/rescore_log.txt "
                 "lists only Kalman/ARIMA/LSTM/BiLSTM for this corridor) -- not computed, not invented.",
    })
    return out


def family_item4_matched():
    out = []
    for phase in c.PHASES:
        traj = P.trajrag_item4_errors(phase)
        base = P.aisacnet_item4_errors(phase)
        out.append(paired_stats(traj, base, f"ShortHorizon/{phase}/AIS-ACNet"))
    for phase in c.PHASES:
        out.append({
            "label": f"ShortHorizon/{phase}/iTransformer(+6 other forecasters)",
            "n": 0,
            "error": "No per-voyage predictions were saved for any modern forecaster, only the "
                     "aggregate ADE/FDE over each forecaster's own (non-matching) test split -- "
                     "a paired test against them is not possible from saved outputs.",
        })
    return out


def family_ablations():
    out = []
    for phase in c.PHASES:
        for ablation in ["no_retrieval", "no_guard", "no_heading"]:
            full, abl = P.ablation_errors("HOU-SHIPCHANNEL", phase, ablation)
            out.append(paired_stats(full, abl, f"Ablation/{phase}/{ablation}"))
    return out


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    f1 = family_houston_main()
    f2 = family_la_main()
    f3 = family_item4_matched()
    f4 = family_ablations()

    holm_correction(f1)
    holm_correction(f2)
    holm_correction(f3)
    holm_correction(f4)

    result = {
        "F1_houston_main": f1,
        "F2_la_main": f2,
        "F3_short_horizon": f3,
        "F4_ablations": f4,
        "methodology": {
            "bootstrap": f"{N_BOOT} resamples, voyages resampled with replacement (paired, not points), "
                         "95% CI = 2.5/97.5 percentiles of the resampled MEAN difference (baseline - TrajRAG). "
                         "Positive = TrajRAG more accurate.",
            "wilcoxon": "paired two-sided Wilcoxon signed-rank test, scipy.stats.wilcoxon, recomputed "
                        "independently from saved per-step predictions (not copied from "
                        "final_significance_all.json, though cross-checked against it for F1 and it matches).",
            "holm": "Holm-Bonferroni step-down correction, alpha=0.05, applied separately within each "
                    "of the four families above (not pooled across families).",
        },
    }
    with open(out_dir / "task2_task3_full_stats.json", "w") as f:
        json.dump(result, f, indent=2)

    def show(entries, name):
        print(f"\n=== {name} ===")
        for e in entries:
            if "error" in e:
                print(f"  {e['label']}: SKIPPED - {e['error']}")
            else:
                sig = "***" if e.get("significant_holm_0.05") else "ns"
                print(f"  {e['label']}: n={e['n']} mean_diff={e['mean_diff_km']} "
                      f"CI={e['mean_diff_ci95']} p={e['wilcoxon_p']:.2e} "
                      f"p_holm={e['p_holm']:.2e} [{sig}]")

    show(f1, "F1 Houston main (n=50)")
    show(f2, "F2 LA/Long Beach main (n=10)")
    show(f3, "F3 Short-horizon matched")
    show(f4, "F4 Ablations (n=20)")
