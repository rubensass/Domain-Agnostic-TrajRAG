"""
Rescores the free (zero-API-cost) baselines on the OFFICIAL 125-per-corridor
test set (test_voyage_selection.json), once on the ORIGINAL
voyage_data_with_minutes_since_start.json and once on this experiment's
voyage_data_corrected_<CORRIDOR>.json (the only difference being the lat/lon
resampling fix in corrected_build_voyages.py). Measures whether the
preprocessing leak materially changes these numbers.

Read-only imports from MARITIME-SPOOFING/methods/*.py and
Engineering Applications of Artificial Intelligence/baselines/knn_maritime/.
Writes: leak_impact_report.json
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", module="statsmodels")

import numpy as np

MARITIME_ROOT = Path(__file__).resolve().parents[3] / "MARITIME-SPOOFING"
THIS_DIR = Path(__file__).parent
KNN_DIR = THIS_DIR.parents[1] / "baselines" / "knn_maritime"

sys.path.insert(0, str(MARITIME_ROOT / "methods"))
sys.path.insert(0, str(KNN_DIR))

from kalman_baseline import kalman_predict_phase, haversine_km  # noqa: E402
from arima_baseline import arima_predict_phase  # noqa: E402
from lstm_baseline import (  # noqa: E402
    seq_to_matrix, compute_norm_stats, train_phase_model, lstm_predict_phase,
)
import knn_baseline_maritime as knn  # noqa: E402

PHASES = ["departure", "transit", "arrival"]
SELECTION_PATH = MARITIME_ROOT / "evaluation" / "results" / "test_voyage_selection.json"


def mean_err(preds, gt) -> float:
    return float(np.mean([haversine_km(gt[i][0], gt[i][1], preds[i][0], preds[i][1]) for i in range(len(preds))]))


def run_kalman_arima(voyage_path: Path, test_ids: set) -> dict:
    """Kalman + ARIMA only: deterministic, no training, so a before/after
    comparison isolates the preprocessing change cleanly. LSTM/BiLSTM are
    skipped for this diagnostic -- full retraining (30 epochs x ~2000+
    sequences) takes on the order of hours per corridor/version, and
    stochastic training noise would confound a same-data-different-fix
    comparison anyway (two LSTM trainings on identical data already differ
    run to run)."""
    with open(voyage_path, "r", encoding="utf-8") as f:
        voyages = json.load(f)

    phase_errs = {phase: {"KALMAN": [], "ARIMA": []} for phase in PHASES}

    for phase in PHASES:
        test_voyages = [v for v in voyages if len(v.get(phase, [])) >= 4 and v["voyage_id"] in test_ids]

        for v in test_voyages:
            seq = v[phase]
            try:
                preds, gt = kalman_predict_phase(seq)
                phase_errs[phase]["KALMAN"].append(mean_err(preds, gt))
            except Exception:
                pass
            try:
                preds, gt = arima_predict_phase(seq)
                phase_errs[phase]["ARIMA"].append(mean_err(preds, gt))
            except Exception:
                pass

    return {
        phase: {method: round(float(np.mean(errs)), 4) if errs else None for method, errs in d.items()}
        for phase, d in phase_errs.items()
    }


def run_knn(voyage_path: Path) -> dict:
    # knn_baseline_maritime.py hardcodes its own VOYAGE_FILE/TEST_SELECTION_FILE
    # module globals; override them in-place (read-only otherwise) to point at
    # our chosen voyage file (Houston only, matching how this baseline is
    # scoped in the manuscript's extended-comparison table).
    knn.VOYAGE_FILE = voyage_path
    entries = knn.load_entries()
    test = [e for e in entries if e["is_test"]]
    remaining = sorted((e for e in entries if not e["is_test"]), key=lambda e: e["timestamp"])
    n_train = int(len(remaining) * 0.85)
    train = remaining[:n_train]

    from collections import defaultdict
    train_index = defaultdict(list)
    for e in train:
        split = e["n"] // 2
        train_index[e["phase"]].append((e["latlon"][:split], e["latlon"][split:]))

    phase_ade = defaultdict(list)
    for e in test:
        split = e["n"] // 2
        prefix, ground_truth = e["latlon"][:split], e["latlon"][split:]
        pred_len = len(ground_truth)
        if pred_len == 0 or not train_index[e["phase"]]:
            continue
        predicted = knn.knn_predict(prefix, pred_len, train_index[e["phase"]])
        errs = [knn.haversine_km(ground_truth[i][0], ground_truth[i][1], predicted[i][0], predicted[i][1])
                for i in range(pred_len)]
        phase_ade[e["phase"]].append(float(np.mean(errs)))

    return {phase: round(float(np.mean(phase_ade[phase])), 4) if phase_ade[phase] else None for phase in PHASES}


def main():
    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        selection = json.load(f)

    corridors = {
        "HOU-SHIPCHANNEL": {
            "original": MARITIME_ROOT / "data" / "HOU-SHIPCHANNEL" / "voyage_data_with_minutes_since_start.json",
            "corrected": THIS_DIR / "voyage_data_corrected_HOU-SHIPCHANNEL.json",
        },
        "LA-LONGBEACH": {
            "original": MARITIME_ROOT / "data" / "LA-LONGBEACH" / "voyage_data_with_minutes_since_start.json",
            "corrected": THIS_DIR / "voyage_data_corrected_LA-LONGBEACH.json",
        },
    }

    report = {}
    for corridor, paths in corridors.items():
        test_ids = {v["voyage_id"] for v in selection[corridor]}
        print(f"\n=== {corridor}: KALMAN/ARIMA ===")
        print("  original...", flush=True)
        orig = run_kalman_arima(paths["original"], test_ids)
        print("  corrected...", flush=True)
        corr = run_kalman_arima(paths["corrected"], test_ids)
        report[corridor] = {"original": orig, "corrected": corr}
        print(json.dumps({"original": orig, "corrected": corr}, indent=2))

    print("\n=== HOU-SHIPCHANNEL: Numeric-kNN ===")
    knn_orig = run_knn(corridors["HOU-SHIPCHANNEL"]["original"])
    knn_corr = run_knn(corridors["HOU-SHIPCHANNEL"]["corrected"])
    report["HOU-SHIPCHANNEL"]["Numeric-kNN"] = {"original": knn_orig, "corrected": knn_corr}
    print(json.dumps({"original": knn_orig, "corrected": knn_corr}, indent=2))

    # percentage-change summary
    summary = {}
    for corridor, d in report.items():
        summary[corridor] = {}
        for phase in PHASES:
            summary[corridor][phase] = {}
            for method in ["KALMAN", "ARIMA"]:
                o = d["original"][phase].get(method)
                c = d["corrected"][phase].get(method)
                if o and c:
                    summary[corridor][phase][method] = {
                        "original_km": o, "corrected_km": c,
                        "pct_change": round(100.0 * (c - o) / o, 2),
                    }
            if "Numeric-kNN" in d:
                o = d["Numeric-kNN"]["original"].get(phase)
                c = d["Numeric-kNN"]["corrected"].get(phase)
                if o and c:
                    summary[corridor][phase]["Numeric-kNN"] = {
                        "original_km": o, "corrected_km": c,
                        "pct_change": round(100.0 * (c - o) / o, 2),
                    }

    out = {"raw": report, "pct_change_summary": summary}
    with open(THIS_DIR / "leak_impact_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n=== PCT CHANGE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {THIS_DIR / 'leak_impact_report.json'}")


if __name__ == "__main__":
    main()
