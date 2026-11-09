"""
Task 1: recover full-precision mean errors for TrajRAG and AIS-ACNet on the
item4 matched short-horizon comparison (the 0.73 vs 0.734 km case,
transit phase), recomputed from saved per-step predicted/ground-truth
positions rather than the rounded mean_error_km field. iTransformer's own
aggregate (ade_km) was already saved at full float precision in
baselines/tslib_maritime/results/iTransformer_*.json -- no rounding issue
there, so it needs no recomputation, only reporting.

No per-voyage breakdown was ever saved for iTransformer (or any other
modern forecaster) -- only the aggregate ade_km/fde_km over its own
99-131-voyage test set. This is reported as a limitation, not invented.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common as c

EAAI = c.EAAI


def recompute_trajrag_item4():
    rows = c.load_item4_matched()
    by_phase = {}
    for row in rows:
        phase = row["phase"]
        errs = [c.haversine_km(p[0], p[1], g[0], g[1])
                for p, g in zip(row["predicted_positions"], row["ground_truth_positions"])]
        voyage_mean = sum(errs) / len(errs) if errs else None
        by_phase.setdefault(phase, []).append(voyage_mean)
    return {
        phase: {
            "n": len(vals),
            "mean_km_full_precision": sum(vals) / len(vals),
            "mean_km_3dp": round(sum(vals) / len(vals), 3),
        }
        for phase, vals in by_phase.items()
    }


def recompute_aisacnet_item4():
    out = {}
    for phase in c.PHASES:
        pts = c.load_ais_acnet_item4_points(phase)
        vals = []
        for vid, d in pts.items():
            errs = [c.haversine_km(p[0], p[1], g[0], g[1])
                    for p, g in zip(d["predicted"], d["ground_truth"])]
            vals.append(sum(errs) / len(errs) if errs else None)
        out[phase] = {
            "n": len(vals),
            "mean_km_full_precision": sum(vals) / len(vals),
            "mean_km_3dp": round(sum(vals) / len(vals), 3),
        }
    return out


def itransformer_aggregate():
    out = {}
    for phase in c.PHASES:
        d = c.load_json(EAAI / "baselines" / "tslib_maritime" / "results" / f"iTransformer_{phase}.json")
        out[phase] = {
            "n_test_voyages_itransformer_own_set": d["test_size"],
            "ade_km_full_precision": d["ade_km"],
            "ade_km_3dp": round(d["ade_km"], 3),
            "note": ("Aggregate ADE only; no per-voyage breakdown was saved for this or any "
                     "other modern forecaster (DLinear/PatchTST/Informer/Autoformer/FEDformer/"
                     "TimesNet), and its test set (n={}) is NOT the same voyage set as TrajRAG's "
                     "matched n={} subset -- so no paired per-voyage comparison or Wilcoxon test "
                     "against iTransformer is possible from saved outputs."
                     .format(d["test_size"], None)),
        }
    return out


if __name__ == "__main__":
    trajrag = recompute_trajrag_item4()
    aisacnet = recompute_aisacnet_item4()
    itrans = itransformer_aggregate()

    for phase in c.PHASES:
        itrans[phase]["note"] = itrans[phase]["note"].replace("n=None", f"n={trajrag[phase]['n']}")

    result = {"TrajRAG": trajrag, "AIS-ACNet": aisacnet, "iTransformer": itrans}
    out_path = Path(__file__).parent / "results" / "task1_precision.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

    print("\n=== Headline check: transit phase (the reported 0.73 vs 0.734 case) ===")
    t = trajrag["transit"]["mean_km_full_precision"]
    i = itrans["transit"]["ade_km_full_precision"]
    a = aisacnet["transit"]["mean_km_full_precision"]
    print(f"TrajRAG   transit: {t:.6f} km -> {t:.3f} km (n={trajrag['transit']['n']})")
    print(f"iTransf.  transit: {i:.6f} km -> {i:.3f} km (n={itrans['transit']['n_test_voyages_itransformer_own_set']}, DIFFERENT voyage set)")
    print(f"AIS-ACNet transit: {a:.6f} km -> {a:.3f} km (n={aisacnet['transit']['n']})")
    print(f"Ordering at 3dp: TrajRAG {'<' if t < i else '>='} iTransformer, TrajRAG {'<' if t < a else '>='} AIS-ACNet")
