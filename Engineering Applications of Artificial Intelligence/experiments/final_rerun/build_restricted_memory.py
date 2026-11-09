"""
Builds a corrected, chronologically-restricted retrieval memory for the
final rerun (supervisor review, items 2 + 5a + 5b combined).

- Preprocessing (item 2): reads voyage_data_corrected_<CORRIDOR>.json from
  experiments/preprocessing_fix/ (causal lat/lon resampling, already built
  and verified -- see that folder's corrected_build_voyages.py).
- Memory contamination + chronology (items 5a/5b): the memory pool is
  restricted to voyage_manifest.json's "memory_pool_voyage_ids" -- the
  chronologically EARLIEST ~50% of non-test voyages per corridor. Every
  voyage usable as a reconstruction query (voyage_manifest.json's
  "main_reconstruction_voyage_ids" / "ablation_voyage_ids" / the item-4
  matched sets) is, by construction, dated AFTER this pool's cutoff, so no
  query can ever retrieve a precedent recorded after itself.

Read-only with respect to MARITIME-SPOOFING/: only imports
PROMPT_PHASE_TEMPLATE / generate_summary / get_embeddings / LLM_MODEL /
EMBED_MODEL from data_pipeline/build_embeddings.py. All outputs are written
under this folder.

Outputs:
  memory_v2_<CORRIDOR>/voyage_summary_restricted.json
  memory_v2_<CORRIDOR>/EMBEDDINGS/faiss_index_<phase>.index
  memory_v2_<CORRIDOR>/EMBEDDINGS/ids_<phase>.json

COST NOTICE: makes real OpenAI calls (summaries + embeddings) for ~50% of
the corpus (~5,000 phase-summaries total, both corridors). Budgeted at
~$27. Do not run without confirming budget first.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np

THIS_DIR = Path(__file__).parent
THESIS_ROOT = THIS_DIR.resolve().parents[2]
MARITIME_PIPELINE = THESIS_ROOT / "MARITIME-SPOOFING" / "data_pipeline"
PREPROCESSING_FIX_DIR = THIS_DIR.parent / "preprocessing_fix"

sys.path.insert(0, str(MARITIME_PIPELINE))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import build_embeddings as be  # noqa: E402  (read-only import of original pipeline)

PHASES = ["departure", "transit", "arrival"]
MAX_WORKERS = 16
CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]


def load_client():
    return be.load_client()


def load_manifest() -> dict:
    with open(THIS_DIR / "voyage_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_corridor_memory(client, corridor: str, pool_ids: set):
    print(f"\n=== {corridor}: building restricted memory ({len(pool_ids)} voyages) ===")
    corrected_path = PREPROCESSING_FIX_DIR / f"voyage_data_corrected_{corridor}.json"
    with open(corrected_path, "r", encoding="utf-8") as f:
        voyages = json.load(f)
    pool_voyages = [v for v in voyages if v["voyage_id"] in pool_ids]
    print(f"  matched {len(pool_voyages)} / {len(pool_ids)} pool voyages in corrected data")

    out_dir = THIS_DIR / f"memory_v2_{corridor}"
    out_dir.mkdir(exist_ok=True)
    checkpoint_path = out_dir / "voyage_summary_restricted.json"

    summary_by_id: dict = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for s in json.load(f):
                summary_by_id[s["voyage_id"]] = s

    tasks = []
    for rec in pool_voyages:
        vid = rec["voyage_id"]
        existing = summary_by_id.get(vid, {})
        vessel_type = rec["vessel_metadata"].get("vessel_name") or "commercial vessel"
        for phase in PHASES:
            data = rec.get(phase) or []
            if not data:
                continue
            if existing.get(phase):
                continue
            tasks.append((vid, phase, data, vessel_type))

    print(f"  {len(tasks)} phase-summaries to generate "
          f"({sum(1 for s in summary_by_id.values() for p in PHASES if s.get(p))} already cached)")

    done = 0

    def _save():
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(list(summary_by_id.values()), f)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(be.generate_summary, client, phase, data, vessel_type): (vid, phase)
            for vid, phase, data, vessel_type in tasks
        }
        for fut in as_completed(futures):
            vid, phase = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                print(f"  WARN: giving up on {vid}/{phase}: {e}")
                text = None
            entry = summary_by_id.setdefault(vid, {"voyage_id": vid})
            entry[phase] = text
            done += 1
            if done % 200 == 0:
                _save()
                print(f"  summarized {done}/{len(tasks)}")

    for v in pool_voyages:
        summary_by_id.setdefault(v["voyage_id"], {"voyage_id": v["voyage_id"]})
    _save()
    print(f"  done: {len(summary_by_id)} voyages summarized")

    emb_dir = out_dir / "EMBEDDINGS"
    emb_dir.mkdir(exist_ok=True)
    summaries = list(summary_by_id.values())
    for phase in PHASES:
        ids, texts = [], []
        for s in summaries:
            if s.get(phase):
                ids.append(s["voyage_id"])
                texts.append(s[phase])
        if not texts:
            continue
        X = be.get_embeddings(client, texts)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, str(emb_dir / f"faiss_index_{phase}.index"))
        with open(emb_dir / f"ids_{phase}.json", "w") as f:
            json.dump(ids, f)
        print(f"  {phase}: indexed {len(ids)} voyages")


def main():
    manifest = load_manifest()
    client = load_client()
    t0 = time.time()
    for corridor in CORRIDORS:
        pool_ids = set(manifest[corridor]["memory_pool_voyage_ids"])
        build_corridor_memory(client, corridor, pool_ids)
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
