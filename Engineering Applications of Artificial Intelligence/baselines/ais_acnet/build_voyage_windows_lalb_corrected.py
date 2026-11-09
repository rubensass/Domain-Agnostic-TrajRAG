"""
Task B: identical to build_voyage_windows_lalb.py except VOYAGE_FILE points
at the corrected (non-leaking) LA/Long Beach data. Writes to
voyage_processed_lalb_corrected/.
"""
from pathlib import Path

import build_voyage_windows_lalb as bvwl

HERE = Path(__file__).resolve().parent
bvwl.VOYAGE_FILE = (
    HERE.parents[1] / "experiments" / "preprocessing_fix" / "voyage_data_corrected_LA-LONGBEACH.json"
)
bvwl.OUT_DIR = HERE / "voyage_processed_lalb_corrected"

if __name__ == "__main__":
    bvwl.main()
