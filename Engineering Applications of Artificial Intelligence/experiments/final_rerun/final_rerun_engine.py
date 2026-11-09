"""
Final-rerun engine: combines three fixes on top of SpeedFixedRAGEngine
(r1_speedfix/patched_engine.py), which already carries the R1 speed-rule fix
and the item-5a test-exclusion retrieval fix:

  - Item 2 (preprocessing leak): voyage_data is read from
    experiments/preprocessing_fix/voyage_data_corrected_<CORRIDOR>.json
    (causal lat/lon resampling) instead of the original MARITIME-SPOOFING
    data file.
  - Item 5b (chronological leak): the retrieval memory (summaries +
    FAISS index) is read from this folder's memory_v2_<CORRIDOR>/, built by
    build_restricted_memory.py from ONLY the chronologically earliest ~50%
    of non-test voyages (voyage_manifest.json's memory_pool_voyage_ids).
    Every voyage used as a reconstruction query in this rerun is, by
    construction, dated after that pool's cutoff.
  - A per-query dynamic safety check is kept on top (inherited from
    _find_similar_voyages_no_test_leak): even though the memory pool is
    already pre-cutoff by construction, retrieval still excludes every
    voyage in the corridor's test set, so this is correct by construction,
    not merely correct by how the data happened to split.

Non-destructive: imports CorridorData from rag_engine.py (read-only); does
not touch MARITIME-SPOOFING/ or config.py. Only _get_corridor is overridden
to point at this experiment's own files.
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
THESIS_ROOT = THIS_DIR.resolve().parents[2]
MARITIME_BACKEND = THESIS_ROOT / "MARITIME-SPOOFING" / "app" / "backend"
PREPROCESSING_FIX_DIR = THIS_DIR.parent / "preprocessing_fix"
R1_SPEEDFIX_DIR = THIS_DIR.parent / "r1_speedfix"

for p in (MARITIME_BACKEND, R1_SPEEDFIX_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rag_engine import CorridorData  # noqa: E402  (read-only)
from patched_engine import SpeedFixedRAGEngine  # noqa: E402

CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]


class FinalRerunEngine(SpeedFixedRAGEngine):
    def _get_corridor(self, corridor: str) -> CorridorData:
        if corridor not in CORRIDORS:
            raise ValueError(f"Unknown corridor '{corridor}'. Valid: {CORRIDORS}")
        if corridor not in self._corridors:
            cfg = {
                "label": corridor,
                "voyage_data": PREPROCESSING_FIX_DIR / f"voyage_data_corrected_{corridor}.json",
                "summaries": THIS_DIR / f"memory_v2_{corridor}" / "voyage_summary_restricted.json",
                "embeddings": THIS_DIR / f"memory_v2_{corridor}" / "EMBEDDINGS",
            }
            cd = CorridorData(corridor, cfg)
            cd.initialize()
            self._corridors[corridor] = cd
        return self._corridors[corridor]


final_rerun_engine = FinalRerunEngine()
