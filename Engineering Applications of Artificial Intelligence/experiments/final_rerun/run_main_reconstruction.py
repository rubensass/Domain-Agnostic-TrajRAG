"""
Final rerun, main reconstruction: 50 Houston + 50 LA/Long Beach voyages
(voyage_manifest.json's main_reconstruction_voyage_ids), deployable speed
rule, full pipeline (no ablation), under the combined fix (corrected
preprocessing + test-excluded + chronologically-restricted retrieval
memory -- see final_rerun_engine.py).

Saves ALL per-step predicted/ground-truth points per voyage/phase, not just
the aggregate mean (every run in this folder does this now).

Budget: ~$20 (measured against the equivalent full-scale run's tokens).
Checkpointed; safe to resume if interrupted.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from final_rerun_engine import final_rerun_engine as engine  # noqa: E402

RESULTS_DIR = THIS_DIR / "results"
CHECKPOINT_PATH = RESULTS_DIR / "main_reconstruction_raw.json"
PHASES = ["departure", "transit", "arrival"]
MAX_WORKERS = 12
K = 6
GPT41_RATE_PER_TOKEN = 2.10 / 1_000_000


def load_manifest():
    with open(THIS_DIR / "voyage_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(corridor, voyage_id, phase):
    result = engine.run_experiment(
        corridor=corridor, voyage_id=voyage_id, phase=phase, k=K,
        with_feedback_loop=True, speed_rule="last_trusted_held",
        ablation=None, alpha=None,
    )
    return result


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    manifest = load_manifest()

    pairs = []
    for corridor in ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]:
        for vid in manifest[corridor]["main_reconstruction_voyage_ids"]:
            pairs.append((corridor, vid))
    print(f"{len(pairs)} voyages loaded")

    for corridor in {c for c, _ in pairs}:
        engine._get_corridor(corridor)
    print("Corridor data + restricted FAISS indexes warmed.")

    results = []
    done_keys = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_keys = {(r["corridor"], r["voyage_id"], r["phase"]) for r in results}
        print(f"Resuming: {len(results)} already completed.")

    tasks = []
    for corridor, voyage_id in pairs:
        cd = engine._get_corridor(corridor)
        voyage = cd.data_by_id.get(voyage_id)
        if not voyage:
            print(f"  WARN: {voyage_id} not found in {corridor}'s loaded data")
            continue
        for phase in PHASES:
            if (corridor, voyage_id, phase) in done_keys:
                continue
            if not voyage.get(phase):
                continue
            tasks.append((corridor, voyage_id, phase))

    total_target = len(results) + len(tasks)
    print(f"{len(tasks)} experiments remaining ({total_target} total)")

    completed = len(results)
    failed = 0
    running_cost = sum(r.get("num_tokens", 0) for r in results) * GPT41_RATE_PER_TOKEN
    t0 = time.time()

    def _save():
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, c, v, p): (c, v, p) for c, v, p in tasks}
        for fut in as_completed(futures):
            corridor, voyage_id, phase = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                running_cost += r.get("num_tokens", 0) * GPT41_RATE_PER_TOKEN
            except Exception as e:
                failed += 1
                print(f"  WARN: {corridor}/{voyage_id}/{phase} failed: {e}")
            completed += 1
            if completed % 10 == 0:
                _save()
                elapsed = time.time() - t0
                print(f"  {completed}/{total_target} done ({failed} failed) "
                      f"-- running cost ~${running_cost:.2f}, {elapsed/60:.1f} min elapsed")

    _save()
    print(f"\nDone. {len(results)} total, {failed} failed. Final running cost ~${running_cost:.2f}")


if __name__ == "__main__":
    main()
