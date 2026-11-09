"""
Final rerun, item 4 (matched short-window comparison): reconstructs TrajRAG
on the EXACT same voyages and EXACT same short horizon the modern
forecasters are scored on (voyage_manifest.json's item4_matched_short_window:
63 departure/4-step, 63 transit/3-step, 53 arrival/4-step -- the forecaster-
eligible voyages intersected with this rerun's chronologically-usable test
set), via _predict_voyage's new max_steps parameter. This gives a true
same-timestamps comparison instead of comparing full-horizon TrajRAG against
a short-horizon forecaster subset.

AIS-ACNet's side of this comparison is handled separately (free rescoring of
its own already-computed predictions on this same voyage/step subset, after
it is retrained on corrected data) -- not part of this script.

Saves ALL per-step predicted/ground-truth points.

Budget: authorized $10-20. HARD STOPS at $19.50 measured running cost (stops
launching new tasks; already-dispatched ones are allowed to finish) --
report back rather than exceed authorization.
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
CHECKPOINT_PATH = RESULTS_DIR / "item4_matched_raw.json"
MAX_WORKERS = 10
K = 6
CORRIDOR = "HOU-SHIPCHANNEL"
GPT41_RATE_PER_TOKEN = 2.10 / 1_000_000
HARD_STOP_USD = 19.50


def load_manifest():
    with open(THIS_DIR / "voyage_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(voyage_id, phase, window):
    result = engine.run_experiment(
        corridor=CORRIDOR, voyage_id=voyage_id, phase=phase, k=K,
        with_feedback_loop=True, speed_rule="last_trusted_held",
        ablation=None, alpha=None, max_steps=window,
    )
    return result


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    manifest = load_manifest()
    item4 = manifest[CORRIDOR]["item4_matched_short_window"]

    cd = engine._get_corridor(CORRIDOR)
    print("Corridor data + restricted FAISS indexes warmed.")

    results = []
    done_keys = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_keys = {(r["voyage_id"], r["phase"]) for r in results}
        print(f"Resuming: {len(results)} already completed.")

    tasks = []
    for phase, d in item4.items():
        window = d["window"]
        for voyage_id in d["voyage_ids"]:
            voyage = cd.data_by_id.get(voyage_id)
            if not voyage or not voyage.get(phase):
                continue
            if (voyage_id, phase) in done_keys:
                continue
            tasks.append((voyage_id, phase, window))

    total_target = len(results) + len(tasks)
    print(f"{len(tasks)} experiments remaining ({total_target} total), hard stop at ${HARD_STOP_USD}")

    completed = len(results)
    failed = 0
    running_cost = sum(r.get("num_tokens", 0) for r in results) * GPT41_RATE_PER_TOKEN
    t0 = time.time()
    stopped_early = False

    def _save():
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        task_iter = iter(tasks)

        def _submit_more(n):
            submitted = 0
            for _ in range(n):
                t = next(task_iter, None)
                if t is None:
                    break
                v, p, w = t
                futures[pool.submit(run_one, v, p, w)] = (v, p, w)
                submitted += 1
            return submitted

        _submit_more(MAX_WORKERS * 2)

        while futures:
            fut = next(as_completed(list(futures.keys())))
            voyage_id, phase, window = futures.pop(fut)
            try:
                r = fut.result()
                results.append(r)
                running_cost += r.get("num_tokens", 0) * GPT41_RATE_PER_TOKEN
            except Exception as e:
                failed += 1
                print(f"  WARN: {voyage_id}/{phase} failed: {e}")
            completed += 1
            if completed % 10 == 0:
                _save()
                elapsed = time.time() - t0
                print(f"  {completed}/{total_target} done ({failed} failed) "
                      f"-- running cost ~${running_cost:.2f}, {elapsed/60:.1f} min elapsed")
            if running_cost >= HARD_STOP_USD:
                stopped_early = True
                break
            _submit_more(1)

    _save()
    status = "STOPPED EARLY (hit budget cap)" if stopped_early else "completed fully"
    print(f"\n{status}. {len(results)} total, {failed} failed. Final running cost ~${running_cost:.2f}")


if __name__ == "__main__":
    main()
