"""
Final rerun, ablations: 35 Houston voyages (voyage_manifest.json's
ablation_voyage_ids, a subset of the 50 used in run_main_reconstruction.py
so ablated results are matched-pair against the full pipeline on identical
voyages) x 3 phases x 3 ablation types (no_retrieval, no_guard, no_heading),
deployable speed rule, under the combined fix.

Saves ALL per-step predicted/ground-truth points per voyage/phase/ablation.

Budget: ~$46 (35 x 3 x 3 = 315 reconstructions). Checkpointed.
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
CHECKPOINT_PATH = RESULTS_DIR / "ablations_raw.json"
PHASES = ["departure", "transit", "arrival"]
ABLATIONS = ["no_retrieval", "no_guard", "no_heading"]
MAX_WORKERS = 12
K = 6
CORRIDOR = "HOU-SHIPCHANNEL"
GPT41_RATE_PER_TOKEN = 2.10 / 1_000_000


def load_manifest():
    with open(THIS_DIR / "voyage_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(voyage_id, phase, ablation):
    result = engine.run_experiment(
        corridor=CORRIDOR, voyage_id=voyage_id, phase=phase, k=K,
        with_feedback_loop=True, speed_rule="last_trusted_held",
        ablation=ablation, alpha=None,
    )
    return result


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    manifest = load_manifest()
    voyage_ids = manifest[CORRIDOR]["ablation_voyage_ids"]
    print(f"{len(voyage_ids)} Houston voyages for ablations")

    cd = engine._get_corridor(CORRIDOR)
    print("Corridor data + restricted FAISS indexes warmed.")

    results = []
    done_keys = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_keys = {(r["voyage_id"], r["phase"], r["ablation"]) for r in results}
        print(f"Resuming: {len(results)} already completed.")

    tasks = []
    for voyage_id in voyage_ids:
        voyage = cd.data_by_id.get(voyage_id)
        if not voyage:
            print(f"  WARN: {voyage_id} not found")
            continue
        for phase in PHASES:
            if not voyage.get(phase):
                continue
            for ablation in ABLATIONS:
                if (voyage_id, phase, ablation) in done_keys:
                    continue
                tasks.append((voyage_id, phase, ablation))

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
        futures = {pool.submit(run_one, v, p, a): (v, p, a) for v, p, a in tasks}
        for fut in as_completed(futures):
            voyage_id, phase, ablation = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                running_cost += r.get("num_tokens", 0) * GPT41_RATE_PER_TOKEN
            except Exception as e:
                failed += 1
                print(f"  WARN: {voyage_id}/{phase}/{ablation} failed: {e}")
            completed += 1
            if completed % 15 == 0:
                _save()
                elapsed = time.time() - t0
                print(f"  {completed}/{total_target} done ({failed} failed) "
                      f"-- running cost ~${running_cost:.2f}, {elapsed/60:.1f} min elapsed")

    _save()
    print(f"\nDone. {len(results)} total, {failed} failed. Final running cost ~${running_cost:.2f}")


if __name__ == "__main__":
    main()
