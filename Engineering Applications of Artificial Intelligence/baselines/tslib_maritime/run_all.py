"""Runs all 7 models x 3 phases = 21 training/evaluation jobs sequentially,
logging progress and collecting a summary table at the end."""
import json
import traceback
from pathlib import Path

from train_tslib_model import train_and_eval
from model_wrapper import SUPPORTED

PHASES = ["departure", "transit", "arrival"]
HERE = Path(__file__).resolve().parent


def main():
    summary = []
    for model_name in SUPPORTED:
        for phase in PHASES:
            print(f"\n{'='*70}\nRUNNING {model_name} / {phase}\n{'='*70}")
            try:
                result = train_and_eval(model_name, phase, epochs=60, patience=15, lr=1e-3)
                summary.append(result)
            except Exception as e:
                print(f"FAILED {model_name}/{phase}: {e}")
                traceback.print_exc()
                summary.append({"model": model_name, "phase": phase, "error": str(e)})

    with open(HERE / "results" / "summary_all.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n\nALL DONE. Summary saved to results/summary_all.json")
    for r in summary:
        if "error" in r:
            print(f"  {r['model']:<12s} {r['phase']:<10s} FAILED: {r['error']}")
        else:
            print(f"  {r['model']:<12s} {r['phase']:<10s} "
                  f"ADE={r['ade_km']:.4f}km  FDE={r['fde_km']:.4f}km")


if __name__ == "__main__":
    main()
