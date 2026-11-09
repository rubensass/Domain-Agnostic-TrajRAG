"""
Task B: identical to build_voyage_windows.py except VOYAGE_FILE points at the
corrected (non-leaking) Houston data. Writes to voyage_processed_corrected/
instead of voyage_processed/ so the original (pre-fix) datasets are untouched.
"""
from pathlib import Path

import build_voyage_windows as bvw

HERE = Path(__file__).resolve().parent
bvw.VOYAGE_FILE = (
    HERE.parents[1] / "experiments" / "preprocessing_fix" / "voyage_data_corrected_HOU-SHIPCHANNEL.json"
)
bvw.OUT_DIR = HERE / "voyage_processed_corrected"

if __name__ == "__main__":
    bvw.main()
