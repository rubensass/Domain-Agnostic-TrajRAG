"""
R3 (action plan, minimum set): maritime ablations on the 125 Houston test
voyages, deployable last_trusted_held speed rule throughout -- no-retrieval
(isolates the value of the maritime index) and no-guard (isolates whether
the kinematic validator drives accuracy or acts mainly as a safety net,
mirroring the aviation ablation). 2 configs x 3 phases x 125 voyages = 750
reconstructions.
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
CHECKPOINT_PATH = RESULTS_DIR / "r3_ablations_raw.json"

PHASES = ["departure", "transit", "arrival"]
ABLATIONS = ["no_retrieval", "no_guard"]
SPEED_RULE = "last_trusted_held"
MAX_WORKERS = 12
K = 6
SAVE_EVERY = 20


def run_one(vid, phase, ablation):
    return speedfixed_engine.run_experiment("HOU-SHIPCHANNEL", vid, phase, k=K,
                                            with_feedback_loop=True, speed_rule=SPEED_RULE,
                                            ablation=ablation)


def main():
    with open(TEST_FILE, encoding="utf-8") as f:
        selection = json.load(f)
    voyage_ids = [e["voyage_id"] for e in selection["HOU-SHIPCHANNEL"]]
    print(f"{len(voyage_ids)} Houston test voyages loaded")

    speedfixed_engine.initialize()
    cd = speedfixed_engine._get_corridor("HOU-SHIPCHANNEL")
    print("Corridor data + FAISS indexes warmed.")

    done_results = []
    done_keys = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            done_results = json.load(f)
        done_keys = {(r["voyage_id"], r["phase"], r["ablation"]) for r in done_results}
        print(f"Resuming: {len(done_results)} already completed.")

    tasks = []
    for vid in voyage_ids:
        voyage = cd.data_by_id.get(vid)
        if not voyage:
            continue
        for phase in PHASES:
            if not voyage.get(phase):
                continue
            for ablation in ABLATIONS:
                if (vid, phase, ablation) in done_keys:
                    continue
                tasks.append((vid, phase, ablation))

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
        futures = {pool.submit(run_one, vid, phase, ablation): (vid, phase, ablation) for vid, phase, ablation in tasks}
        for fut in as_completed(futures):
            vid, phase, ablation = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                failed += 1
                print(f"  WARN: {vid}/{phase}/{ablation} failed: {e}", flush=True)
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
