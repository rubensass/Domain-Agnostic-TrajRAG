"""
Two zero-cost (no OpenAI calls, no training) maritime baselines, added per a
supervisor review of the TrajRAG manuscript:

  - "stay": predicts the last trusted (pre-blackout) position for every
    withheld step (item 6 -- checks whether TrajRAG's arrival advantage
    merely reflects already-stationary vessels).
  - "headingdr": heading-aware dead reckoning -- starts from the last
    trusted position, holds speed at the last trusted SOG for the whole
    blackout (TrajRAG's own deployed last_trusted_held speed rule), and at
    each step projects forward using that step's live trusted gyrocompass
    heading (item 1 -- checks whether TrajRAG's live-heading access alone,
    without any LLM or retrieval, explains its accuracy).

Read-only with respect to MARITIME-SPOOFING/ (original thesis code): this
script only reads data/ and evaluation/results/ from there. All outputs are
written under this folder.

LIMITATION (headingdr): this is pure heading+speed dead reckoning with NO
ocean-current compensation, unlike the Kalman baseline (methods/kalman_baseline.py)
which does inject current drift. Currents are a few tenths of a km/h in this
corpus (see sec_instantiation.tex environmental fields), so the omission is
expected to be a second-order effect relative to heading/speed error, but it
means any current-driven drift during the blackout is not modeled here and
will show up as extra error specifically attributable to that choice, not to
heading-DR's core mechanism.
"""

import json
import math
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[3]
MARITIME_ROOT = THESIS_ROOT / "MARITIME-SPOOFING"
OUT_DIR = Path(__file__).parent / "results"

CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]
PHASES = ["departure", "transit", "arrival"]
IDLE_SOG_KNOTS = 1.0  # matches data_pipeline/build_voyages.py's IDLE_SOG_KNOTS


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def project(lat, lon, bearing_deg, dist_km):
    R = 6371.0
    d = dist_km / R
    lat1, lon1, b = map(math.radians, [lat, lon, bearing_deg])
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def load_test_ids(corridor: str) -> set:
    path = MARITIME_ROOT / "evaluation" / "results" / "test_voyage_selection.json"
    with open(path, "r", encoding="utf-8") as f:
        sel = json.load(f)
    return {v["voyage_id"] for v in sel[corridor]}


def load_voyages(corridor: str) -> dict:
    path = MARITIME_ROOT / "data" / corridor / "voyage_data_with_minutes_since_start.json"
    with open(path, "r", encoding="utf-8") as f:
        voyages = json.load(f)
    return {v["voyage_id"]: v for v in voyages}


def split_phase(phase_data: list):
    n = len(phase_data)
    mid = n // 2
    observed = phase_data[:mid]
    future = phase_data[mid:]
    last_verified = observed[-1] if observed else future[0]
    return observed, future, last_verified


def stay_baseline(phase_data: list):
    """Predict the last trusted position for every withheld step. Returns
    (mean_error_km, is_moving) or None if there's nothing to score."""
    observed, future, last_verified = split_phase(phase_data)
    if not future:
        return None
    lat0, lon0 = last_verified["latitude"], last_verified["longitude"]
    errs = [haversine_km(lat0, lon0, p["latitude"], p["longitude"]) for p in future]
    sog0 = last_verified.get("sog") or 0.0
    is_moving = sog0 >= IDLE_SOG_KNOTS
    return sum(errs) / len(errs), is_moving


def headingdr_baseline(phase_data: list):
    """Heading-aware dead reckoning: speed frozen at last trusted SOG, bearing
    taken from each step's live trusted heading, autoregressive (compounds
    from its own prior projection, never snaps back to ground truth). No
    current compensation (see module docstring)."""
    observed, future, last_verified = split_phase(phase_data)
    if not future:
        return None

    cur_lat, cur_lon = last_verified["latitude"], last_verified["longitude"]
    speed_knots = last_verified.get("sog") or 0.0
    prev_t = last_verified["minutes_since_start"]
    last_heading = last_verified.get("heading")  # fallback if a future step's heading is missing

    errs = []
    for p in future:
        dt_hours = (p["minutes_since_start"] - prev_t) / 60.0
        bearing = p.get("heading")
        if bearing is None:
            bearing = last_heading  # carry forward last known trusted heading
        if bearing is None:
            bearing = 0.0  # no heading ever seen yet; arbitrary but logged as a known limitation
        else:
            last_heading = bearing

        dist_km = speed_knots * 1.852 * max(dt_hours, 0.0)
        cur_lat, cur_lon = project(cur_lat, cur_lon, bearing, dist_km)
        errs.append(haversine_km(cur_lat, cur_lon, p["latitude"], p["longitude"]))
        prev_t = p["minutes_since_start"]

    return sum(errs) / len(errs)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for corridor in CORRIDORS:
        test_ids = load_test_ids(corridor)
        voyages = load_voyages(corridor)
        summary[corridor] = {}

        for phase in PHASES:
            stay_results = {}
            stay_moving_errs = []
            stay_stationary_errs = []
            headingdr_results = {}
            headingdr_missing_heading_voyages = 0

            for vid in test_ids:
                v = voyages.get(vid)
                if not v or not v.get(phase):
                    continue
                phase_data = v[phase]
                if len(phase_data) < 2:
                    continue

                stay_out = stay_baseline(phase_data)
                if stay_out is not None:
                    err, is_moving = stay_out
                    stay_results[vid] = err
                    (stay_moving_errs if is_moving else stay_stationary_errs).append(err)

                any_heading = any(p.get("heading") is not None for p in phase_data)
                if not any_heading:
                    headingdr_missing_heading_voyages += 1
                hdr_err = headingdr_baseline(phase_data)
                if hdr_err is not None:
                    headingdr_results[vid] = hdr_err

            with open(OUT_DIR / f"stay_{corridor}_{phase}.json", "w", encoding="utf-8") as f:
                json.dump(stay_results, f, indent=2)
            with open(OUT_DIR / f"headingdr_{corridor}_{phase}.json", "w", encoding="utf-8") as f:
                json.dump(headingdr_results, f, indent=2)

            def stats(errs):
                if not errs:
                    return {"n": 0, "mean_km": None, "median_km": None, "max_km": None}
                s = sorted(errs)
                n = len(s)
                median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
                return {"n": n, "mean_km": round(sum(s) / n, 4), "median_km": round(median, 4), "max_km": round(max(s), 4)}

            all_stay_errs = list(stay_results.values())
            summary[corridor][phase] = {
                "stay": {
                    "all": stats(all_stay_errs),
                    "moving": stats(stay_moving_errs),
                    "stationary": stats(stay_stationary_errs),
                },
                "headingdr": {
                    "all": stats(list(headingdr_results.values())),
                    "voyages_with_no_heading_at_all": headingdr_missing_heading_voyages,
                },
            }
            print(f"{corridor}/{phase}: stay n={summary[corridor][phase]['stay']['all']['n']} "
                  f"mean={summary[corridor][phase]['stay']['all']['mean_km']}km | "
                  f"headingdr n={summary[corridor][phase]['headingdr']['all']['n']} "
                  f"mean={summary[corridor][phase]['headingdr']['all']['mean_km']}km")

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    run()
