"""
Phase 3: reshape enriched voyages into the aircraft-study's phase-keyed schema,
generate per-phase textual state summaries with an LLM, embed them, and build
one FAISS index per phase per corridor.

Mirrors GPS-SPOOFING-APP/backend/rag_engine.py's PROMPT_PHASE_TEMPLATE /
_generate_summary / _get_embeddings (read-only reference; this is a fresh
implementation for the vessel domain, not an edit of that file).

Output layout (parallels THESIS/data/<ROUTE>/...):
  data/<CORRIDOR>/voyage_data_with_minutes_since_start.json
  data/<CORRIDOR>/voyage_summary_separated_into_phases.json
  data/<CORRIDOR>/EMBEDDINGS/faiss_index_<phase>.index
  data/<CORRIDOR>/EMBEDDINGS/ids_<phase>.json

COST NOTICE: this script makes real OpenAI API calls (chat completions for
summaries + embeddings for retrieval). Do not run against the full dataset
without confirming budget/API key first.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # picks up THESIS_ROOT/.env (OPENAI_API_KEY / OPENAI_API_KEY_2)

MAX_WORKERS = 16

DATA_ROOT = Path(__file__).parent.parent / "data"
CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]
PHASES = ["departure", "transit", "arrival"]

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"

PROMPT_PHASE_TEMPLATE = """
You are given a segment of AIS data corresponding to the {phase_name} phase of
a commercial vessel's voyage through a coastal corridor. Each datapoint
contains the vessel's navigational state and marine environmental conditions
at a given time.

Keys per datapoint:
- latitude, longitude, heading, cog (course over ground), sog (speed over
  ground, knots), minutes_since_start
- wave_height (m), wave_direction, wave_period (s), swell_wave_height (m),
  ocean_current_velocity (km/h), ocean_current_direction,
  sea_surface_temperature (C), wind_speed_10m (km/h), wind_direction_10m

Write a continuous, expert-level narrative describing this phase of the
voyage chronologically. For each datapoint:
1. Reference time via minutes_since_start.
2. Describe position (lat/lon), motion (heading, course over ground, speed).
3. Include the marine environmental context (sea state, swell, current, wind,
   sea-surface temperature).
4. Mention relevant geographic features based on coordinates (e.g. channel,
   anchorage, breakwater, open water).
5. Use smooth transitions between datapoints.
6. Start with the vessel type: {vessel_type}

Write as a natural, continuous, professional description -- not bullet points.

Voyage phase data:
{phase_data}

Return the summary without any meta-text.
"""


def load_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY_2") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Set OPENAI_API_KEY (or OPENAI_API_KEY_2) before running this script.")
    return OpenAI(api_key=key)


def reshape_to_phase_keyed(voyages: list) -> list:
    """[{vessel_metadata, points:[{...,'phase':...}]}] -> [{vessel_metadata, departure:[...], transit:[...], arrival:[...]}]"""
    out = []
    for v in voyages:
        rec = {"vessel_metadata": v["vessel_metadata"], "voyage_id": f"{v['vessel_metadata']['mmsi']}_{v['start_time']}"}
        for phase in PHASES:
            rec[phase] = [p for p in v["points"] if p.get("phase") == phase]
        out.append(rec)
    return out


def generate_summary(client: OpenAI, phase_name: str, phase_data: list, vessel_type: str) -> str:
    prompt = PROMPT_PHASE_TEMPLATE.format(
        phase_name=phase_name, phase_data=json.dumps(phase_data), vessel_type=vessel_type,
    )
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"summary generation failed after 3 attempts: {last_err}")


def get_embeddings(client: OpenAI, texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    X = np.vstack(vecs)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def build_corridor(client: OpenAI, name: str):
    print(f"\n=== {name} ===")
    in_path = DATA_ROOT / name / "voyages_enriched.json"
    with open(in_path, "r", encoding="utf-8") as f:
        voyages = json.load(f)

    phase_keyed = reshape_to_phase_keyed(voyages)
    with open(DATA_ROOT / name / "voyage_data_with_minutes_since_start.json", "w", encoding="utf-8") as f:
        json.dump(phase_keyed, f)

    # Resumable checkpoint: if a previous run crashed partway through, skip
    # (voyage_id, phase) pairs we already paid for and summarized.
    checkpoint_path = DATA_ROOT / name / "voyage_summary_separated_into_phases.json"
    summary_by_id: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for s in json.load(f):
                summary_by_id[s["voyage_id"]] = s

    tasks = []
    for rec in phase_keyed:
        vid = rec["voyage_id"]
        existing = summary_by_id.get(vid, {})
        for phase in PHASES:
            data = rec[phase]
            if not data:
                continue
            if existing.get(phase):
                continue  # already summarized in a prior run
            vessel_type = rec["vessel_metadata"].get("vessel_name") or "commercial vessel"
            tasks.append((vid, phase, data, vessel_type))

    print(f"  {len(tasks)} phase-summaries to generate "
          f"({sum(1 for r in phase_keyed for p in PHASES if r[p])} total, "
          f"{sum(1 for r in phase_keyed for p in PHASES if r[p]) - len(tasks)} already cached)")

    done = 0
    lock_save_every = 100

    def _save_checkpoint():
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(list(summary_by_id.values()), f)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(generate_summary, client, phase, data, vessel_type): (vid, phase)
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
            if done % lock_save_every == 0:
                _save_checkpoint()
                print(f"  summarized {done}/{len(tasks)} phase-entries")

    # Ensure every voyage has an entry (even ones fully skipped/cached above)
    for rec in phase_keyed:
        summary_by_id.setdefault(rec["voyage_id"], {"voyage_id": rec["voyage_id"]})
    summaries = list(summary_by_id.values())
    _save_checkpoint()
    print(f"  done: {len(summaries)} voyages summarized")

    emb_dir = DATA_ROOT / name / "EMBEDDINGS"
    emb_dir.mkdir(exist_ok=True)
    for phase in PHASES:
        ids, texts = [], []
        for s in summaries:
            if s.get(phase):
                ids.append(s["voyage_id"])
                texts.append(s[phase])
        if not texts:
            continue
        X = get_embeddings(client, texts)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, str(emb_dir / f"faiss_index_{phase}.index"))
        with open(emb_dir / f"ids_{phase}.json", "w") as f:
            json.dump(ids, f)
        print(f"  {phase}: indexed {len(ids)} voyages")


def main():
    client = load_client()
    for name in CORRIDORS:
        build_corridor(client, name)


if __name__ == "__main__":
    main()
