"""
Cheap pilot (3 Houston test voyages x 3 phases x 3 speed rules = up to 27
reconstructions) to validate the patched engine and get a real per-call cost
reading before committing to the full 125-voyage run.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv
MARITIME_ENV = HERE.parents[2] / "MARITIME-SPOOFING" / "app" / "backend" / ".env"
load_dotenv(MARITIME_ENV if MARITIME_ENV.exists() else None)
load_dotenv()

from patched_engine import speedfixed_engine  # noqa: E402

TEST_FILE = HERE.parents[2] / "MARITIME-SPOOFING" / "evaluation" / "results" / "test_voyage_selection.json"
PHASES = ["departure", "transit", "arrival"]
SPEED_RULES = ["oracle", "last_trusted_held", "aviation_inferred"]
N_PILOT_VOYAGES = 3

with open(TEST_FILE, encoding="utf-8") as f:
    selection = json.load(f)
pilot_ids = [e["voyage_id"] for e in selection["HOU-SHIPCHANNEL"][:N_PILOT_VOYAGES]]
print(f"Pilot voyages: {pilot_ids}")

speedfixed_engine.initialize()
cd = speedfixed_engine._get_corridor("HOU-SHIPCHANNEL")

results = []
t0 = time.time()
total_tokens = 0
for vid in pilot_ids:
    voyage = cd.data_by_id.get(vid)
    if not voyage:
        print(f"  SKIP {vid}: not found")
        continue
    for phase in PHASES:
        if not voyage.get(phase):
            continue
        for rule in SPEED_RULES:
            try:
                r = speedfixed_engine.run_experiment("HOU-SHIPCHANNEL", vid, phase, k=6,
                                                     with_feedback_loop=True, speed_rule=rule)
                results.append(r)
                total_tokens += r["num_tokens"]
                print(f"  {vid}/{phase}/{rule}: error={r['mean_error_km']:.3f}km  "
                      f"tokens={r['num_tokens']}  time={r['elapsed_sec']:.1f}s", flush=True)
            except Exception as e:
                print(f"  FAILED {vid}/{phase}/{rule}: {e}", flush=True)
            with open(HERE / "results" / "pilot_results.json", "w") as f:
                json.dump(results, f, indent=2)

elapsed = time.time() - t0
print(f"\nTotal: {len(results)} reconstructions, {total_tokens} tokens, {elapsed:.1f}s")

# GPT-4.1 pricing: $2.00/1M input, $8.00/1M output (approx blended estimate
# using total_tokens as a conservative proxy skewed toward input-heavy calls)
est_cost = total_tokens / 1_000_000 * 3.0
print(f"Rough cost estimate (blended $3/1M token proxy): ${est_cost:.3f}")
print(f"Per-reconstruction average: {total_tokens/len(results):.0f} tokens, "
      f"${est_cost/len(results):.4f}")

out_path = HERE / "results" / "pilot_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved -> {out_path}")
