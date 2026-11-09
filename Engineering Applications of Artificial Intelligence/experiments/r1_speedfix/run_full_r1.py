"""
R1 full run: 125 Houston test voyages x 3 phases x 2 deployable speed-rule
variants (last_trusted_held = R1b, aviation_inferred = R1a). The "oracle"
(original, leaky) rule is NOT re-run here -- those numbers are already the
ones published in the current draft (Table 6), kept only as a reference row.

Parallelized with a thread pool (same pattern as
MARITIME-SPOOFING/evaluation/run_trajrag_eval.py), checkpointed to disk.
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
CHECKPOINT_PATH = RESULTS_DIR / "r1_full_raw.json"

PHASES = ["departure", "transit", "arrival"]
SPEED_RULES = ["last_trusted_held", "aviation_inferred"]
MAX_WORKERS = 12
K = 6
SAVE_EVERY = 20


def run_one(vid, phase, rule):
    r = speedfixed_engine.run_experiment("HOU-SHIPCHANNEL", vid, phase, k=K,
                                         with_feedback_loop=True, speed_rule=rule)
    return r


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
        done_keys = {(r["voyage_id"], r["phase"], r["speed_rule"]) for r in done_results}
        print(f"Resuming: {len(done_results)} already completed.")

    tasks = []
    for vid in voyage_ids:
        voyage = cd.data_by_id.get(vid)
        if not voyage:
            continue
        for phase in PHASES:
            if not voyage.get(phase):
                continue
            for rule in SPEED_RULES:
                if (vid, phase, rule) in done_keys:
                    continue
                tasks.append((vid, phase, rule))

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
        futures = {pool.submit(run_one, vid, phase, rule): (vid, phase, rule) for vid, phase, rule in tasks}
        for fut in as_completed(futures):
            vid, phase, rule = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                failed += 1
                print(f"  WARN: {vid}/{phase}/{rule} failed: {e}", flush=True)
            completed += 1
            if completed % SAVE_EVERY == 0:
                _save()
                elapsed = time.time() - t0
                rate = (completed - len(done_results)) / elapsed if elapsed > 0 else 0
                remaining = (total_target - completed) / rate if rate > 0 else float("nan")
                total_tokens = sum(r.get("num_tokens", 0) for r in results)
                est_cost = total_tokens / 1_000_000 * 3.0
                print(f"  {completed}/{total_target} done ({failed} failed) "
                      f"-- {rate*60:.1f}/min, ~{remaining/60:.0f} min remaining, "
                      f"running cost estimate ${est_cost:.2f}", flush=True)

    _save()
    total_tokens = sum(r.get("num_tokens", 0) for r in results)
    print(f"\nDone. {len(results)} total, {failed} failed. "
          f"Total tokens: {total_tokens}, est. cost ${total_tokens/1_000_000*3.0:.2f}")


if __name__ == "__main__":
    main()
