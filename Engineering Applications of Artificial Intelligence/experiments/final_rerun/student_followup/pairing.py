"""
Builds matched (voyage_id -> trajrag_error_km, baseline_error_km) pairs for
every comparison family needed by tasks 2 and 3, from already-saved results
only. No new reconstructions, no LLM calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common as c


def _mean_err(predicted, ground_truth):
    errs = [c.haversine_km(p[0], p[1], g[0], g[1]) for p, g in zip(predicted, ground_truth)]
    return sum(errs) / len(errs) if errs else None


def trajrag_main_errors(corridor, phase):
    """voyage_id -> TrajRAG mean error (full precision), main n=50/10 set."""
    rows = c.load_main_reconstruction()
    out = {}
    for r in rows:
        if r["corridor"] == corridor and r["phase"] == phase:
            out[r["voyage_id"]] = _mean_err(r["predicted_positions"], r["ground_truth_positions"])
    return out


def mean_retrieval_main_errors(corridor, phase):
    """voyage_id -> Mean-Retrieval mean error, computed from the SAME main rows'
    mean_similar_positions (the retrieval-only baseline uses TrajRAG's own
    retrieved neighbours, averaged, with no LLM)."""
    rows = c.load_main_reconstruction()
    out = {}
    for r in rows:
        if r["corridor"] == corridor and r["phase"] == phase and r["mean_similar_positions"]:
            out[r["voyage_id"]] = _mean_err(r["mean_similar_positions"], r["ground_truth_positions"])
    return out


def classical_baseline_errors(corridor, method, phase):
    """voyage_id -> baseline mean error, for Kalman/ARIMA/LSTM/BiLSTM/Numeric-kNN."""
    detail = c.load_classical_detail(corridor)
    if method not in detail:
        return {}
    out = {}
    for row in detail[method][phase]:
        out[row["voyage_id"]] = _mean_err(row["predicted"], row["ground_truth"])
    return out


def aisacnet_main_errors(corridor, phase):
    """voyage_id -> AIS-ACNet mean error, full125 set (caller filters to matched ids)."""
    return c.load_ais_acnet_per_voyage(corridor, phase)


def aisacnet_item4_errors(phase):
    pts = c.load_ais_acnet_item4_points(phase)
    return {vid: _mean_err(d["predicted"], d["ground_truth"]) for vid, d in pts.items()}


def trajrag_item4_errors(phase):
    rows = c.load_item4_matched()
    out = {}
    for r in rows:
        if r["phase"] == phase:
            out[r["voyage_id"]] = _mean_err(r["predicted_positions"], r["ground_truth_positions"])
    return out


def ablation_errors(corridor, phase, ablation):
    """voyage_id -> mean error for the full pipeline and for a named ablation,
    restricted to the 20-voyage ablation subset. ablations_raw.json contains
    only the 3 ablated configs (no ablation=None rows); the matched full-
    pipeline comparator is the same 20 voyages' rows in
    main_reconstruction_raw.json (verified: ablation_voyage_ids subset of
    main_reconstruction_voyage_ids, voyage_manifest.json)."""
    abl_rows = c.load_ablations()
    abl = {}
    for r in abl_rows:
        if r["corridor"] == corridor and r["phase"] == phase and r["ablation"] == ablation:
            abl[r["voyage_id"]] = _mean_err(r["predicted_positions"], r["ground_truth_positions"])

    main_rows = c.load_main_reconstruction()
    full = {}
    for r in main_rows:
        if r["corridor"] == corridor and r["phase"] == phase and r["voyage_id"] in abl:
            full[r["voyage_id"]] = _mean_err(r["predicted_positions"], r["ground_truth_positions"])
    return full, abl


def matched_pairs(dict_a, dict_b):
    """Return (ids, a_vals, b_vals) for voyage_ids present in both, sorted for determinism."""
    ids = sorted(set(dict_a) & set(dict_b))
    return ids, [dict_a[i] for i in ids], [dict_b[i] for i in ids]
