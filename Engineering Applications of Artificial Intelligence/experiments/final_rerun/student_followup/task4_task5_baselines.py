"""
Task 4: constant-position baseline ("stays at last trusted position").
Task 5: heading-aware dead-reckoning baseline (trusted heading + last
trusted SOG held fixed, geographic destination-point propagation).

Both are pure arithmetic over the corrected voyage_data files already used
by final_rerun_engine.py -- no LLM calls, no re-running TrajRAG or
AIS-ACNet. Evaluated on the identical voyages/timestamps as the main
TrajRAG reconstruction (results/main_reconstruction_raw.json) and the
item4 matched short-horizon subset, for both ports.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common as c


def constant_position_predict(last_lat, last_lon, spoofed):
    return [[last_lat, last_lon, p["minutes_since_start"]] for p in spoofed]


def heading_dr_predict(last_lat, last_lon, last_sog_kn, last_t_min, last_heading, spoofed):
    """Dead reckoning: speed fixed at last trusted SOG (same convention as
    the deployable last_trusted_held guard rule); heading read fresh at
    each step from the trusted (non-GNSS) signal; distance = speed * dt;
    position propagated from the PREVIOUS ESTIMATED position via the
    geographic destination-point formula (not naive lat/lon addition)."""
    preds = []
    cur_lat, cur_lon = last_lat, last_lon
    prev_t = last_t_min
    cur_heading = last_heading  # fallback if a step's heading is ever missing
    missing_heading_steps = 0
    for p in spoofed:
        t_min = p["minutes_since_start"]
        dt_hours = (t_min - prev_t) / 60.0
        heading = p.get("heading")
        if heading is None:
            missing_heading_steps += 1
        else:
            cur_heading = heading
        dist_km = last_sog_kn * 1.852 * dt_hours  # knots -> km/h
        if cur_heading is None:
            # no trusted heading has ever been observed; cannot propagate a
            # bearing, so hold position (reported, not silently guessed)
            new_lat, new_lon = cur_lat, cur_lon
        else:
            new_lat, new_lon = c.destination_point(cur_lat, cur_lon, cur_heading, dist_km)
        preds.append([new_lat, new_lon, t_min])
        cur_lat, cur_lon = new_lat, new_lon
        prev_t = t_min
    return preds, missing_heading_steps


def run_for_rowset(rows, label):
    out = []
    total_missing_heading = 0
    for row in rows:
        corridor, voyage_id, phase = row["corridor"], row["voyage_id"], row["phase"]
        observed, spoofed = c.get_observed_and_spoofed(corridor, voyage_id, phase, row.get("max_steps"))
        if not observed:
            # no trusted prefix at all (phase starts at step 0 of the voyage) -- cannot
            # define "last trusted position"; skip and report, do not fabricate
            out.append({
                "corridor": corridor, "voyage_id": voyage_id, "phase": phase,
                "skipped_no_trusted_prefix": True,
            })
            continue
        last = observed[-1]
        last_lat, last_lon = last["latitude"], last["longitude"]
        last_sog = last.get("sog") or 0.0
        last_heading = last.get("heading")
        last_t_min = last["minutes_since_start"]

        gt = row["ground_truth_positions"]

        const_pred = constant_position_predict(last_lat, last_lon, spoofed)
        const_err = [c.haversine_km(p[0], p[1], g[0], g[1]) for p, g in zip(const_pred, gt)]
        const_mean = sum(const_err) / len(const_err) if const_err else None

        hdr_pred, n_missing = heading_dr_predict(last_lat, last_lon, last_sog, last_t_min, last_heading, spoofed)
        total_missing_heading += n_missing
        hdr_err = [c.haversine_km(p[0], p[1], g[0], g[1]) for p, g in zip(hdr_pred, gt)]
        hdr_mean = sum(hdr_err) / len(hdr_err) if hdr_err else None

        out.append({
            "corridor": corridor, "voyage_id": voyage_id, "phase": phase,
            "max_steps": row.get("max_steps"),
            "last_trusted_sog_kn": last_sog, "last_trusted_heading_deg": last_heading,
            "constant_position": {
                "predicted_positions": const_pred, "ground_truth_positions": gt,
                "per_step_error_km": const_err, "mean_error_km": const_mean,
            },
            "heading_dr": {
                "predicted_positions": hdr_pred, "ground_truth_positions": gt,
                "per_step_error_km": hdr_err, "mean_error_km": hdr_mean,
                "missing_heading_steps": n_missing,
            },
        })
    print(f"[{label}] {len(out)} voyages, total missing-heading steps across all = {total_missing_heading}")
    return out


def summarize(results, group_keys):
    """group_keys: function(row)->tuple key. Returns dict key(joined) -> {n, const_mean, hdr_mean}."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r.get("skipped_no_trusted_prefix"):
            continue
        groups[group_keys(r)].append(r)
    summary = {}
    for key, rs in groups.items():
        cm = [r["constant_position"]["mean_error_km"] for r in rs]
        hm = [r["heading_dr"]["mean_error_km"] for r in rs]
        summary["/".join(key)] = {
            "n": len(rs),
            "constant_position_mean_km": round(sum(cm) / len(cm), 4),
            "heading_dr_mean_km": round(sum(hm) / len(hm), 4),
        }
    return summary


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    main_rows = c.load_main_reconstruction()
    main_results = run_for_rowset(main_rows, "main (n=50 HOU / n=10 LA per phase)")
    with open(out_dir / "task4_task5_main.json", "w") as f:
        json.dump(main_results, f, indent=2)

    item4_rows = c.load_item4_matched()
    item4_results = run_for_rowset(item4_rows, "item4 matched short-horizon")
    with open(out_dir / "task4_task5_item4.json", "w") as f:
        json.dump(item4_results, f, indent=2)

    main_summary = summarize(main_results, lambda r: (r["corridor"], r["phase"]))
    item4_summary = summarize(item4_results, lambda r: (r["phase"],))

    with open(out_dir / "task4_task5_summary.json", "w") as f:
        json.dump({"main": main_summary, "item4_matched": item4_summary}, f, indent=2)

    print(json.dumps({"main": main_summary, "item4_matched": item4_summary}, indent=2))
