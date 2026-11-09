"""
Task 6: representative and difficult case plots.

Selection rule (stated explicitly, per the task): for a given
port x phase, rank the 50 (Houston) or 10 (LA/Long Beach) main-set voyages
by TrajRAG's own mean reconstruction error; the "representative" case is the
voyage closest to the MEDIAN of that ranking, the "difficult" case is the
voyage closest to the 90th percentile. Both are plotted with the trusted
prefix, the blackout onset, the true withheld trajectory, and both TrajRAG's
and AIS-ACNet's reconstructions.

Guard-intervention markers are NOT added: the saved records only contain
each step's FINAL accepted position, never the raw pre-clamp LLM proposal
for the voyages actually used here (the raw proposal is only observable by
re-running the LLM, which is out of scope; the no_guard ablation runs a
*different*, independently-sampled autoregressive sequence on a different
voyage subset, so its predicted_positions cannot be used to infer where the
guard clamped *this* voyage's *main* reconstruction -- per the task's own
caveat, "final accepted positions alone may be insufficient," and here nothing
else was saved). This omission is stated in the figure caption, not patched
over.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import common as c
import pairing as P

FIG_DIR = Path(__file__).parent / "figures"
RES_DIR = Path(__file__).parent / "results"


def pick_cases(corridor, phase):
    traj_err = P.trajrag_main_errors(corridor, phase)
    ids = sorted(traj_err, key=lambda k: traj_err[k])
    errs = np.array([traj_err[i] for i in ids])
    n = len(ids)

    median_target = np.median(errs)
    p90_target = np.percentile(errs, 90)

    rep_idx = int(np.argmin(np.abs(errs - median_target)))
    diff_idx = int(np.argmin(np.abs(errs - p90_target)))

    return {
        "representative": {"voyage_id": ids[rep_idx], "trajrag_error_km": float(errs[rep_idx]),
                            "rank": rep_idx + 1, "n": n, "median_of_group": float(median_target)},
        "difficult": {"voyage_id": ids[diff_idx], "trajrag_error_km": float(errs[diff_idx]),
                      "rank": diff_idx + 1, "n": n, "p90_of_group": float(p90_target)},
    }


def plot_case(corridor, phase, voyage_id, case_label, ax):
    observed, spoofed = c.get_observed_and_spoofed(corridor, voyage_id, phase)
    rows = c.load_main_reconstruction()
    row = next(r for r in rows if r["corridor"] == corridor and r["phase"] == phase and r["voyage_id"] == voyage_id)
    pred_traj = row["predicted_positions"]
    gt = row["ground_truth_positions"]

    ais_pts = c.load_ais_acnet_points(corridor, phase)
    ais = ais_pts.get(voyage_id)

    obs_lat = [p["latitude"] for p in observed]
    obs_lon = [p["longitude"] for p in observed]
    gt_lat = [p[0] for p in gt]
    gt_lon = [p[1] for p in gt]
    pred_lat = [p[0] for p in pred_traj]
    pred_lon = [p[1] for p in pred_traj]

    ax.plot(obs_lon, obs_lat, "-o", color="#555555", markersize=3, linewidth=1.5, label="Trusted prefix")
    ax.plot(obs_lon[-1], obs_lat[-1], "s", color="black", markersize=8, zorder=5, label="Blackout onset")
    ax.plot(gt_lon, gt_lat, "-o", color="#1b9e77", markersize=3, linewidth=1.8, label="True withheld track")
    ax.plot(pred_lon, pred_lat, "--^", color="#d95f02", markersize=4, linewidth=1.6, label="TrajRAG reconstruction")

    if ais is not None:
        ais_lat = [p[0] for p in ais["predicted"]]
        ais_lon = [p[1] for p in ais["predicted"]]
        ax.plot(ais_lon, ais_lat, "--d", color="#7570b3", markersize=4, linewidth=1.6, label="AIS-ACNet reconstruction")

    traj_err = np.mean([c.haversine_km(p[0], p[1], g[0], g[1]) for p, g in zip(pred_traj, gt)])
    blackout_min = gt[-1][2] - row.get("max_steps", None) if False else (gt[-1][2] - observed[-1]["minutes_since_start"])
    title = f"{corridor}, {phase} -- {case_label}\nTrajRAG MHE={traj_err:.2f} km, blackout={blackout_min:.0f} min"
    if ais is not None:
        ais_err = np.mean([c.haversine_km(p[0], p[1], g[0], g[1]) for p, g in zip(ais["predicted"], gt)])
        title += f", AIS-ACNet MHE={ais_err:.2f} km"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=7)
    return {"voyage_id": voyage_id, "trajrag_mhe_km": float(traj_err),
            "aisacnet_mhe_km": float(ais_err) if ais is not None else None,
            "blackout_minutes": float(blackout_min)}


def make_figure(corridor, phase, out_name):
    cases = pick_cases(corridor, phase)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    meta = {"corridor": corridor, "phase": phase, "selection": cases, "plotted": {}}
    meta["plotted"]["representative"] = plot_case(corridor, phase, cases["representative"]["voyage_id"],
                                                    "representative (median)", axes[0])
    meta["plotted"]["difficult"] = plot_case(corridor, phase, cases["difficult"]["voyage_id"],
                                              "difficult (90th pct.)", axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / out_name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return meta


if __name__ == "__main__":
    RES_DIR.mkdir(exist_ok=True)
    all_meta = {}
    all_meta["HOU-SHIPCHANNEL_arrival"] = make_figure("HOU-SHIPCHANNEL", "arrival", "case_houston_arrival.png")
    all_meta["LA-LONGBEACH_transit"] = make_figure("LA-LONGBEACH", "transit", "case_la_transit.png")

    with open(RES_DIR / "task6_case_selection.json", "w") as f:
        json.dump(all_meta, f, indent=2)
    print(json.dumps(all_meta, indent=2))
