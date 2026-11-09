"""
Task A: rescore classical/numeric baselines on CORRECTED data, restricted to
the exact voyage sets TrajRAG was reconstructed on in this final rerun
(voyage_manifest.json's main_reconstruction_voyage_ids: 50 Houston, 10
LA/Long Beach). Saves full per-step predicted/ground-truth points, not just
aggregate MHE. Read-only imports from MARITIME-SPOOFING/methods/*.py and the
knn baseline.
"""
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).parent
FINAL_RERUN_DIR = THIS_DIR.parent
EXPERIMENTS_DIR = FINAL_RERUN_DIR.parent
EAAI_DIR = EXPERIMENTS_DIR.parent
THESIS_ROOT = EAAI_DIR.parent
MARITIME_METHODS = THESIS_ROOT / "MARITIME-SPOOFING" / "methods"
KNN_DIR = EAAI_DIR / "baselines" / "knn_maritime"
PREPROC_DIR = EXPERIMENTS_DIR / "preprocessing_fix"

sys.path.insert(0, str(MARITIME_METHODS))
sys.path.insert(0, str(KNN_DIR))

from kalman_baseline import kalman_predict_phase, haversine_km  # noqa: E402
from arima_baseline import arima_predict_phase  # noqa: E402
from lstm_baseline import (  # noqa: E402
    seq_to_matrix, compute_norm_stats, train_phase_model, lstm_predict_phase,
)
import knn_baseline_maritime as knn  # noqa: E402

PHASES = ["departure", "transit", "arrival"]
MANIFEST_PATH = FINAL_RERUN_DIR / "voyage_manifest.json"
TEST_SELECTION_PATH = THESIS_ROOT / "MARITIME-SPOOFING" / "evaluation" / "results" / "test_voyage_selection.json"
OUT_DIR = THIS_DIR


def mean_err(preds, gt):
    return float(np.mean([haversine_km(gt[i][0], gt[i][1], preds[i][0], preds[i][1]) for i in range(len(preds))]))


def pts(arr):
    return [[float(p[0]), float(p[1])] for p in arr]


def run_corridor(corridor, voyage_ids, methods):
    with open(PREPROC_DIR / f"voyage_data_corrected_{corridor}.json", "r", encoding="utf-8") as f:
        all_voyages = {v["voyage_id"]: v for v in json.load(f)}
    with open(TEST_SELECTION_PATH, "r", encoding="utf-8") as f:
        sel = json.load(f)
    test_ids = {v["voyage_id"] for v in sel[corridor]}
    non_test = [v for vid, v in all_voyages.items() if vid not in test_ids]

    results = {m: {p: {} for p in PHASES} for m in methods}
    detail = {m: {p: [] for p in PHASES} for m in methods}

    trained_lstm = {}
    if "LSTM" in methods or "BiLSTM" in methods:
        for phase in PHASES:
            train_seqs = [v[phase] for v in non_test if len(v.get(phase, [])) >= 4]
            train_mats = [seq_to_matrix(s) for s in train_seqs]
            means, stds = compute_norm_stats(train_mats)
            if "LSTM" in methods:
                trained_lstm[("LSTM", phase)] = (train_phase_model(train_mats, means, stds, epochs=30, bidirectional=False), means, stds)
            if "BiLSTM" in methods:
                trained_lstm[("BiLSTM", phase)] = (train_phase_model(train_mats, means, stds, epochs=30, bidirectional=True), means, stds)

    knn_index = None
    if "Numeric-kNN" in methods:
        knn.VOYAGE_FILE = PREPROC_DIR / f"voyage_data_corrected_{corridor}.json"
        entries = knn.load_entries()
        remaining = sorted((e for e in entries if not e["is_test"]), key=lambda e: e["timestamp"])
        n_train = int(len(remaining) * 0.85)
        train = remaining[:n_train]
        from collections import defaultdict
        knn_index = defaultdict(list)
        for e in train:
            split = e["n"] // 2
            knn_index[e["phase"]].append((e["latlon"][:split], e["latlon"][split:]))
        entries_by_id = {(e["voyage_id"], e["phase"]): e for e in entries}

    for vid in voyage_ids:
        v = all_voyages.get(vid)
        if not v:
            print(f"  WARN missing {vid}")
            continue
        for phase in PHASES:
            seq = v.get(phase, [])
            if len(seq) < 4:
                continue
            if "Kalman" in methods:
                try:
                    preds, gt = kalman_predict_phase(seq)
                    results["Kalman"][phase][vid] = mean_err(preds, gt)
                    detail["Kalman"][phase].append({"voyage_id": vid, "predicted": pts(preds), "ground_truth": pts(gt)})
                except Exception as e:
                    print(f"  Kalman fail {vid}/{phase}: {e}")
            if "ARIMA" in methods:
                try:
                    preds, gt = arima_predict_phase(seq)
                    results["ARIMA"][phase][vid] = mean_err(preds, gt)
                    detail["ARIMA"][phase].append({"voyage_id": vid, "predicted": pts(preds), "ground_truth": pts(gt)})
                except Exception as e:
                    print(f"  ARIMA fail {vid}/{phase}: {e}")
            for name in ("LSTM", "BiLSTM"):
                if name in methods:
                    try:
                        model, means, stds = trained_lstm[(name, phase)]
                        preds, gt = lstm_predict_phase(model, seq, means, stds)
                        results[name][phase][vid] = mean_err(preds, gt)
                        detail[name][phase].append({"voyage_id": vid, "predicted": pts(preds), "ground_truth": pts(gt)})
                    except Exception as e:
                        print(f"  {name} fail {vid}/{phase}: {e}")
            if "Numeric-kNN" in methods:
                e = entries_by_id.get((vid, phase))
                if e:
                    split = e["n"] // 2
                    prefix, gtv = e["latlon"][:split], e["latlon"][split:]
                    if gtv and knn_index[phase]:
                        predv = knn.knn_predict(prefix, len(gtv), knn_index[phase])
                        err = float(np.mean([knn.haversine_km(gtv[i][0], gtv[i][1], predv[i][0], predv[i][1]) for i in range(len(gtv))]))
                        results["Numeric-kNN"][phase][vid] = err
                        detail["Numeric-kNN"][phase].append({"voyage_id": vid, "predicted": pts(predv), "ground_truth": pts(gtv)})

    return results, detail


def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    jobs = {
        "HOU-SHIPCHANNEL": ["Kalman", "ARIMA", "LSTM", "BiLSTM", "Numeric-kNN"],
        "LA-LONGBEACH": ["Kalman", "ARIMA", "LSTM", "BiLSTM"],
    }
    summary = {}
    for corridor, methods in jobs.items():
        voyage_ids = manifest[corridor]["main_reconstruction_voyage_ids"]
        print(f"=== {corridor}: {len(voyage_ids)} voyages, methods={methods} ===")
        results, detail = run_corridor(corridor, voyage_ids, methods)

        with open(OUT_DIR / f"detail_{corridor}.json", "w", encoding="utf-8") as f:
            json.dump(detail, f)

        summary[corridor] = {}
        for m in methods:
            summary[corridor][m] = {}
            for phase in PHASES:
                errs = list(results[m][phase].values())
                summary[corridor][m][phase] = {
                    "n": len(errs),
                    "mean_km": round(float(np.mean(errs)), 4) if errs else None,
                }
        with open(OUT_DIR / f"aggregate_{corridor}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(summary[corridor], indent=2))

    with open(OUT_DIR / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
