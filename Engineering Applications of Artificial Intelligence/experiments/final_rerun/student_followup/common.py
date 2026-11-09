"""
Shared utilities for the student follow-up task list (supervisor's 6-item
list, 2026-11-01). Read-only with respect to every existing result file:
this module only loads already-saved JSON and the corrected voyage_data
files, and performs arithmetic (haversine distance, bootstrap, Wilcoxon,
geographic destination-point projection). No LLM calls anywhere.

Paths are relative to this file so the module runs from any cwd.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
FINAL_RERUN = HERE.parent
EXPERIMENTS = FINAL_RERUN.parent
EAAI = EXPERIMENTS.parent

R_EARTH_KM = 6371.0  # matches rag_engine.py's _haversine_km / _bearing_deg exactly

CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]
PHASES = ["departure", "transit", "arrival"]


def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R_EARTH_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def destination_point(lat1, lon1, bearing_deg, dist_km):
    """Spherical direct/forward geodetic problem: move dist_km from
    (lat1, lon1) along compass bearing bearing_deg (0=N, 90=E clockwise).
    Same formula and Earth radius as the kinematic guard's own
    _apply_kinematic_validation clamp (patched_engine.py), for methodological
    consistency with the rest of the codebase."""
    d = dist_km / R_EARTH_KM
    lat1r, lon1r, b = map(math.radians, [lat1, lon1, bearing_deg])
    lat2r = math.asin(math.sin(lat1r) * math.cos(d) + math.cos(lat1r) * math.sin(d) * math.cos(b))
    lon2r = lon1r + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(lat1r),
        math.cos(d) - math.sin(lat1r) * math.sin(lat2r),
    )
    return math.degrees(lat2r), math.degrees(lon2r)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Saved experiment outputs (TrajRAG, AIS-ACNet, classical/recurrent baselines)
# ---------------------------------------------------------------------------

def load_main_reconstruction():
    """180 rows: TrajRAG full pipeline, ablation=None, max_steps=None,
    50 Houston + 10 LA/Long Beach voyages x 3 phases."""
    rows = load_json(FINAL_RERUN / "results" / "main_reconstruction_raw.json")
    return [r for r in rows if r["ablation"] is None and r["max_steps"] is None]


def load_ablations():
    return load_json(FINAL_RERUN / "results" / "ablations_raw.json")


def load_item4_matched():
    """179 rows: TrajRAG on the modern-forecasters' exact short window/subset."""
    return load_json(FINAL_RERUN / "results" / "item4_matched_raw.json")


def load_ais_acnet_per_voyage(corridor, phase):
    """dict voyage_id -> mean_error_km, full125 test set (n=125)."""
    return load_json(FINAL_RERUN / "ais_acnet_corrected" / f"per_voyage_{corridor}_{phase}.json")


def load_ais_acnet_points(corridor, phase):
    """dict voyage_id -> {predicted: [[lat,lon],...], ground_truth: [[lat,lon],...]}, full125."""
    return load_json(FINAL_RERUN / "ais_acnet_corrected" / f"points_{corridor}_{phase}.json")


def load_ais_acnet_item4_points(phase):
    """dict voyage_id -> {predicted, ground_truth}, matched short-window subset, Houston only."""
    return load_json(FINAL_RERUN / "ais_acnet_corrected" / f"item4_points_{phase}.json")


def load_classical_detail(corridor):
    """dict method -> phase -> list of {voyage_id, predicted, ground_truth}."""
    return load_json(FINAL_RERUN / "baseline_rescore" / f"detail_{corridor}.json")


def load_significance_all():
    return load_json(FINAL_RERUN / "final_significance_all.json")


# ---------------------------------------------------------------------------
# Raw corrected voyage data (for the constant-position and heading-DR
# baselines): same file final_rerun_engine.py itself reads, loaded directly
# with no engine/LLM involved.
# ---------------------------------------------------------------------------

_VOYAGE_CACHE = {}


def load_voyage_data(corridor):
    if corridor not in _VOYAGE_CACHE:
        path = EXPERIMENTS / "preprocessing_fix" / f"voyage_data_corrected_{corridor}.json"
        data = load_json(path)
        _VOYAGE_CACHE[corridor] = {v["voyage_id"]: v for v in data}
    return _VOYAGE_CACHE[corridor]


def get_observed_and_spoofed(corridor, voyage_id, phase, max_steps=None):
    """Exact replica of patched_engine.py:_predict_voyage's halving logic
    (same spoof_index = N // 2 split), read-only, no model call."""
    voyages = load_voyage_data(corridor)
    phase_data = voyages[voyage_id][phase]
    n = len(phase_data)
    spoof_index = n // 2
    observed = phase_data[:spoof_index]
    spoofed = phase_data[spoof_index:]
    if max_steps is not None:
        spoofed = spoofed[:max_steps]
    return observed, spoofed
