"""
R6 (action plan): phase-calibrated kinematic tolerance for departure and
arrival (transit already uses alpha=1.2, which Section 6.7's own
calibration found to be well-matched to transit's displacement-ratio
distribution, so it is not re-run here). Uses the exact 99th-percentile
values computed in that calibration: alpha=5.08 (departure), alpha=1.74
(arrival). 125 Houston voyages x 2 phases = 250 reconstructions, deployable
last_trusted_held speed rule throughout.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv
load_dotenv()

from patched_engine import speedfixed_engine  # noqa: E402

TEST_FILE = HERE.parents[2] / "MARITIME-SPOOFING" / "evaluation" / "results" / "test_voyage_selection.json"
RESULTS_DIR = HERE / "results"
CHECKPOINT_PATH = RESULTS_DIR / "r6_alpha_raw.json"

PHASE_ALPHAS = {"departure": 5.08, "arrival": 1.74}
SPEED_RULE = "last_trusted_held"
MAX_WORKERS = 12
K = 6
SAVE_EVERY = 20


def run_one(vid, phase):
    alpha = PHASE_ALPHAS[phase]
    return speedfixed_engine.run_experiment("HOU-SHIPCHANNEL", vid, phase, k=K,
                                            with_feedback_loop=True, speed_rule=SPEED_RULE,
                                            alpha=alpha)


def main():
    with open(TEST_FILE, encoding="utf-8") as f:
        selection = json.load(f)
    voyage_ids = [e["voyage_id"] for e in selection["HOU-SHIPCHANNEL"]]
    print(f"{len(voyage_ids)} Houston test voyages loaded, phases={list(PHASE_ALPHAS)}")

    speedfixed_engine.initialize()
    cd = speedfixed_engine._get_corridor("HOU-SHIPCHANNEL")
    print("Corridor data + FAISS indexes warmed.")

    done_results = []
    done_keys = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            done_results = json.load(f)
        done_keys = {(r["voyage_id"], r["phase"]) for r in done_results}
        print(f"Resuming: {len(done_results)} already completed.")

    tasks = []
    for vid in voyage_ids:
        voyage = cd.data_by_id.get(vid)
        if not voyage:
            continue
        for phase in PHASE_ALPHAS:
            if not voyage.get(phase):
                continue
            if (vid, phase) in done_keys:
                continue
            tasks.append((vid, phase))

    total_target = len(done_results) + len(tasks)
    print(f"{len(tasks)} remaining ({total_target} total)")

    results = list(done_results)
    completed = len(results)
    failed = 0
    t0 = time.time()

    def _save():
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, vid, phase): (vid, phase) for vid, phase in tasks}
        for fut in as_completed(futures):
            vid, phase = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                failed += 1
                print(f"  WARN: {vid}/{phase} failed: {e}", flush=True)
            completed += 1
            if completed % SAVE_EVERY == 0:
                _save()
                elapsed = time.time() - t0
                rate = (completed - len(done_results)) / elapsed if elapsed > 0 else 0
                remaining = (total_target - completed) / rate if rate > 0 else float("nan")
                total_tokens = sum(r.get("num_tokens", 0) for r in results)
                est_cost = total_tokens / 1_000_000 * 2.1
                print(f"  {completed}/{total_target} done ({failed} failed) "
                      f"-- {rate*60:.1f}/min, ~{remaining/60:.0f} min remaining, "
                      f"running cost estimate ${est_cost:.2f}", flush=True)

    _save()
    total_tokens = sum(r.get("num_tokens", 0) for r in results)
    print(f"\nDone. {len(results)} total, {failed} failed. "
          f"Total tokens: {total_tokens}, est. cost ${total_tokens/1_000_000*2.1:.2f}")


if __name__ == "__main__":
    main()
