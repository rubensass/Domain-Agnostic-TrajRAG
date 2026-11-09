# TrajRAG — Maritime Instantiation (EAAI journal extension)

Code, data, and results for the maritime half of:

> **Retrieval-augmented large language models for training-free trajectory
> reconstruction under satellite navigation spoofing in aviation and
> maritime surveillance**
> Ruben Sasson, Yuval Felendler, Yuval Elovici, Asaf Shabtai —
> *Engineering Applications of Artificial Intelligence* (submitted)

TrajRAG reconstructs an aircraft's or a vessel's position during a GNSS
("GPS") spoofing blackout using a **frozen** large language model, conditioned
on retrieved precedent trajectories and a kinematic feasibility guard — no
model training, no fine-tuning. The conference version of this method
(`paper/sec_intro.tex`, `\citep{felendler2026trajrag}`) covered aviation only.
This repository is the code, the exact data, and the exact results for
**extending it to shipping**: the retrieval memory, prompt templates, trusted
signal and kinematic constants change (the "domain adapter"); the engine
itself (retrieval → LLM proposal → guard → feedback loop) does not.

This repository corresponds **only** to the maritime extension folder of the
authors' working tree (`Engineering Applications of Artificial
Intelligence/`), plus the small, pruned set of pre-existing files it reads
from (`MARITIME-SPOOFING/`, vendored read-only — see
[Provenance](#provenance-what-is-and-isnt-in-this-repo)). It does not include
the aviation codebase, which is a separate, earlier project.

## Table of contents

1. [Repository layout](#repository-layout)
2. [Setup](#setup)
3. [The dataset](#the-dataset)
4. [The method](#the-method-trajrag-engine)
5. [Reproducing the paper's results, step by step](#reproducing-the-papers-results-step-by-step)
6. [Baselines](#baselines)
7. [Results: where they are and how to read them](#results-where-they-are-and-how-to-read-them)
8. [File and function reference](#file-and-function-reference)
9. [Provenance: what is and isn't in this repo](#provenance-what-is-and-isnt-in-this-repo)
10. [Citation](#citation)

---

## Repository layout

```
TrajRAG-maritime/
├── Engineering Applications of Artificial Intelligence/
│   ├── baselines/              every baseline method compared against TrajRAG
│   │   ├── ais_acnet/            trained CNN baseline (Shin et al., same journal)
│   │   ├── knn_maritime/         non-parametric retrieval-only baseline
│   │   ├── tslib_maritime/       7 generic deep forecasters (DLinear, PatchTST, ...)
│   │   └── simple_baselines/     constant-position / heading-dead-reckoning baselines
│   ├── experiments/             all scripts that actually produce the paper's numbers
│   │   ├── preprocessing_fix/    the corrected, causal-resampling dataset build
│   │   ├── r1_speedfix/          the deployable kinematic speed-rule patch
│   │   └── final_rerun/          the FINAL, corrected pipeline + every paper result
│   ├── paper/                   the submitted manuscript (LaTeX source + compiled PDF)
│   └── presentation_summary/    a 3-page summary PDF of the maritime + aviation results
└── MARITIME-SPOOFING/           pruned, read-only vendor copy (see Provenance)
    ├── app/backend/               the TrajRAG engine itself (rag_engine.py)
    ├── data_pipeline/              scripts that build voyages from raw AIS + weather
    ├── data/                       the two corridors' ORIGINAL (pre-correction) voyage files
    ├── methods/                    Kalman / ARIMA / LSTM baseline implementations
    └── evaluation/results/         the 125-voyage-per-corridor test-set definition
```

Every script in `Engineering Applications of Artificial Intelligence/` reads
and writes paths relative to its own location using `Path(__file__)`
arithmetic that expects `MARITIME-SPOOFING/` to be its **sibling** directory
(both directly under this repo's root) — exactly as laid out above. You do
not need to move anything to run the code; just clone this repo as-is.

## Setup

```bash
git clone <this-repo-url> TrajRAG-maritime
cd TrajRAG-maritime
python3 -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Python 3.10+ is required (3.10 specifically for `faiss`/`utm`-adjacent
dependencies in a couple of the AIS-ACNet scripts; everything else runs on
3.10–3.12).

### Where to put your OpenAI key

Every script that calls the LLM reads its key the same way
(`MARITIME-SPOOFING/app/backend/config.py:get_openai_key()`):

```python
def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY_2") or os.getenv("OPENAI_API_KEY")
    ...
```

Create a `.env` file at the **repo root** (loaded via `python-dotenv`):

```
OPENAI_API_KEY=sk-...
```

(`OPENAI_API_KEY_2` is just a fallback the original authors used to split
quota across two keys — a single `OPENAI_API_KEY` is all you need.)

**Nothing in this repo requires an API key except the scripts that actually
make retrieval-conditioned LLM calls** — i.e. anything that imports
`rag_engine.py` and calls `run_experiment(...)` (the `experiments/final_rerun/
run_*.py` scripts). Everything else — all the baselines, all the statistics,
all the figures — runs on the already-saved JSON results in this repo and
needs no key at all. If you only want to inspect/replot the paper's numbers,
skip straight to [§7](#results-where-they-are-and-how-to-read-them).

### Running with a different LLM

- **A different OpenAI model** (e.g. a newer GPT release): edit `LLM_MODEL`
  and/or `EMBED_MODEL` in `MARITIME-SPOOFING/app/backend/config.py`. This is
  the only place the model name is set; every script imports it from there.
  The paper's maritime results use `gpt-4.1` (aviation uses `gpt-5.1`,
  Section 5.5 of the paper measures the effect of this choice directly).
- **Azure OpenAI or any OpenAI-compatible proxy**: pass `base_url=...` to the
  `OpenAI(...)` client constructed at `rag_engine.py:194`
  (`self.client = OpenAI(api_key=get_openai_key())`).
- **A non-OpenAI vendor (Anthropic, etc.)**: the two call sites that would
  need to be swapped for a different SDK are `rag_engine.py:281` (the
  one-off phase summary) and `rag_engine.py:338` (the per-step position
  proposal, both currently `self.client.chat.completions.create(...)`). The
  prompt-construction logic around both calls (§"TrajRAG's textual
  representation" below) is vendor-agnostic; only the call itself and the
  response-parsing need replacing.

## The dataset

### What it is

Two U.S. port corridors, AIS (Automatic Identification System) data from the
**NOAA Marine Cadastre** public archive, enriched with marine weather from
**Open-Meteo Marine**:

| Corridor | Label | Test voyages (full set) |
|---|---|---|
| `HOU-SHIPCHANNEL` | Galveston Bay → Port of Houston (Houston Ship Channel) | 125 |
| `LA-LONGBEACH` | San Pedro Bay approach → Port of LA / Long Beach | 125 |

Each voyage is split into three phases — `departure`, `transit`, `arrival` —
each resampled to a fixed 5-minute grid. A blackout is simulated as the
second (chronological) half of each phase: the model sees the first half
(the "trusted prefix") and must reconstruct the second half (withheld ground
truth) using nothing but the trusted, GNSS-independent signals.

Per-point fields (see any file under
`MARITIME-SPOOFING/data/<corridor>/voyage_data_with_minutes_since_start.json`
for the raw schema): `latitude, longitude` (withheld during blackout),
`heading` (gyrocompass, **trusted** — independent of GNSS), `cog, sog`
(course/speed over ground, GPS-derived and therefore **untrusted** during a
real blackout, though `sog` is held fixed from its last trusted value as the
model's reference speed), plus marine environmental context (`wave_height`,
`swell_wave_height`, `ocean_current_velocity`, `sea_surface_temperature`,
`wind_speed_10m`, ...).

### How it was built

`MARITIME-SPOOFING/data_pipeline/`:
- `download_ais.py` — pulls raw AIS records for a corridor's bounding box from
  the NOAA archive.
- `build_voyages.py` — segments the raw AIS stream into individual voyages,
  labels the three phases from SOG/heading dynamics, resamples to the fixed
  5-minute grid, joins marine weather.
- `enrich_weather.py` — fetches and merges Open-Meteo Marine data.
- `build_embeddings.py` — turns each phase into a natural-language narrative
  via one LLM call (§"The method" below), embeds it with
  `text-embedding-3-large`, builds a per-phase/per-corridor FAISS index. This
  is TrajRAG's offline retrieval-memory build step.

### A correctness fix that changes the dataset used throughout the paper

Two bugs were found in a supervisor review and fixed; **every number in the
paper uses the corrected data**, not the original pipeline's output:

1. **Preprocessing leak.** The original 5-minute resampling grid
   interpolated `latitude`/`longitude` with a method that could read an AIS
   report recorded *after* the grid timestamp (non-causal). Fixed in
   `Engineering Applications of Artificial Intelligence/experiments/
   preprocessing_fix/corrected_build_voyages.py`, which forward-fills instead
   — the same causal rule already used for every other field. Output:
   `experiments/preprocessing_fix/voyage_data_corrected_<CORRIDOR>.json`
   (this is **the dataset** every result in the paper is computed against).
2. **Retrieval-memory leak.** The original retrieval memory excluded only
   the exact query voyage from the corpus, not the other 124 test voyages,
   and had no chronological cutoff, so a reconstruction could retrieve
   another test voyage, or a voyage recorded *after* the query, as a
   "precedent" — both are impossible for a genuinely real-time system. Fixed
   by restricting the memory to the chronologically earliest ~50% of
   non-test voyages per corridor and excluding the full test set outright:
   `experiments/final_rerun/build_restricted_memory.py` builds
   `experiments/final_rerun/memory_v2_<CORRIDOR>/` (summaries + FAISS index)
   from only `voyage_manifest.json`'s `memory_pool_voyage_ids`;
   `experiments/final_rerun/final_rerun_engine.py` subclasses the engine to
   read from this restricted memory instead of the original.

### Test-set definition

`MARITIME-SPOOFING/evaluation/results/test_voyage_selection.json` — the
original 125-voyage-per-corridor held-out test set. Because fix (2) above
required rebuilding the retrieval memory from scratch and re-running every
reconstruction, and a full 125-per-corridor LLM rerun was not affordable, the
paper's **headline TrajRAG numbers use a smaller common test set**: 50
Houston voyages and 10 LA/Long Beach voyages, every one of them dated after
its corridor's memory cutoff
(`experiments/final_rerun/voyage_manifest.json`'s
`main_reconstruction_voyage_ids`). Every other method (all baselines) is
rescored on this identical subset for a matched, paired comparison; AIS-ACNet
is additionally reported on the full 125-voyage set as a check that the
smaller sample isn't anomalous (it isn't — see `tab:extended` in the paper).

## The method: TrajRAG engine

`MARITIME-SPOOFING/app/backend/rag_engine.py` (vendored read-only, see
[Provenance](#provenance-what-is-and-isnt-in-this-repo)) is the engine every
experiment in this repo calls into. One engine, two things happen at offline
build time and inference time:

- **Offline** (`CorridorData.initialize()`): load the corridor's voyage
  data, cached phase narratives, and FAISS indices.
- **Online, per reconstructed step** (`RAGEngine._predict_voyage()`):
  1. Embed the verified/already-reconstructed prefix's narrative.
  2. Retrieve the $K$ most similar historical phase narratives (same phase,
     excluding the test set and anything chronologically later — see above).
  3. Prompt the LLM with the prefix narrative + live trusted heading signal +
     retrieved precedents + prediction history (feedback loop) to propose a
     `{"lat": ..., "lon": ...}`.
  4. Clamp the proposal with a kinematic feasibility guard: max distance =
     `last_trusted_sog_knots × 1.852 × Δt_hours × 1.2` from the last accepted
     position (`_apply_kinematic_validation`).
  5. Accept the (possibly clamped) position, append to history, repeat.

`Engineering Applications of Artificial Intelligence/experiments/
r1_speedfix/patched_engine.py` subclasses this engine
(`SpeedFixedRAGEngine(RAGEngine)`) to add: the deployable speed rule (hold
the last trusted SOG fixed for the whole blackout, rather than leaking
future SOG — see `sec_discussion.tex`, Threats to Validity (i)); the
retrieval-memory leak fix above; and three ablation modes
(`no_retrieval`, `no_guard`, `no_heading`). `experiments/final_rerun/
final_rerun_engine.py` subclasses that again
(`FinalRerunEngine(SpeedFixedRAGEngine)`) only to point `_get_corridor` at
the corrected voyage data and restricted memory. **No file in
`MARITIME-SPOOFING/` is ever edited** — every fix is a new subclass that
overrides one method; this is deliberate, so the original, pre-correction
pipeline remains runnable and auditable side by side with the corrected one.

## Reproducing the paper's results, step by step

All of this has already been run once; the output JSON for every step below
is already committed in this repo (§7). To **re-run** any step (requires an
OpenAI key and will incur API cost):

| # | What | Script | Needs LLM? |
|---|---|---|---|
| 1 | Build the corrected dataset | `experiments/preprocessing_fix/corrected_build_voyages.py` | no |
| 2 | Build the restricted retrieval memory | `experiments/final_rerun/build_restricted_memory.py` | yes (embeddings only) |
| 3 | Main TrajRAG reconstruction (50 Houston + 10 LA/LB voyages × 3 phases) | `experiments/final_rerun/run_main_reconstruction.py` | yes |
| 4 | Maritime ablations (no_retrieval / no_guard / no_heading, 20 voyages × 3 phases) | `experiments/final_rerun/run_ablations.py` | yes |
| 5 | Short-horizon matched comparison vs. the 7 generic forecasters | `experiments/final_rerun/run_item4_matched.py` | yes |
| 6 | Rescore classical/recurrent/retrieval-only baselines on the corrected data | `experiments/final_rerun/baseline_rescore/rescore_baselines.py` | no |
| 7 | Retrain AIS-ACNet on the corrected data | `baselines/ais_acnet/train_corrected_all.py` | no (local training) |
| 8 | Recalibrate the kinematic guard tolerance | `experiments/final_rerun/calibration_corrected.py` | no |
| 9 | Precision/bootstrap/Holm-correction statistics, 2 new baselines, case-study plots | `experiments/final_rerun/student_followup/*.py` | no |

Each `run_*.py` is checkpointed (safe to interrupt/resume) and prints a
running API-cost estimate. `voyage_manifest.json` fixes every voyage
selection with `seed=42` so a rerun draws the identical voyage set.

## Baselines

| Baseline | Trains on maritime data? | Code | Idea |
|---|---|---|---|
| **AIS-ACNet** | yes, per phase/corridor | `baselines/ais_acnet/` | Dilated-CNN, two parallel encoders (position vs. SOG/COG) with gated fusion. Published in this same journal (Shin et al., EAAI 2024); the strongest literature baseline here. Original code: `external/AIS-ACNet` is **not vendored** — see [Provenance](#provenance-what-is-and-isnt-in-this-repo). |
| **Kalman** | no | `MARITIME-SPOOFING/methods/kalman_baseline.py` | Current-aware constant-velocity filter. |
| **ARIMA** | per-voyage fit | `MARITIME-SPOOFING/methods/arima_baseline.py` | Per-voyage ARIMA fit on the trusted prefix. |
| **LSTM / BiLSTM** | yes | `MARITIME-SPOOFING/methods/lstm_baseline.py` | Recurrent sequence model, trained per phase. |
| **Mean-Retrieval** | no | *(no separate script)* — computed for free as a byproduct of every TrajRAG run (`mean_similar_positions` field), since it just averages the same retrieved neighbours TrajRAG itself retrieves, without the LLM. |
| **Numeric-kNN** | no | `baselines/knn_maritime/knn_baseline_maritime.py` | Finds the 5 nearest training prefixes by raw coordinate distance, averages their continuations. |
| **DLinear / PatchTST / iTransformer / Informer / Autoformer / FEDformer / TimesNet** | yes | `baselines/tslib_maritime/` | 7 generic deep forecasters from `thuml/Time-Series-Library` (**not vendored**, see Provenance), evaluated on a small, fixed exact-fit window per phase (not the full blackout half — a disclosed limitation, see `sec_results.tex` §"Comparison with a trained ship-specific network"). |
| **Constant position** | no | `baselines/simple_baselines/simple_baselines.py` (full-125-set version) and `experiments/final_rerun/student_followup/task4_task5_baselines.py` (matched-main-set version used in the paper's Table) | Predicts the vessel stays at its last trusted position for the whole blackout. |
| **Heading dead reckoning** | no | same two files as above | Propagates from the last trusted position using the last trusted SOG (held fixed) and the *live* trusted heading at each step, via the geographic destination-point formula (not planar lat/lon addition). |

## Results: where they are and how to read them

Every number in the paper traces to one of these files. All are plain JSON;
`n` is always the number of voyages in that cell, every distance is km.

| Paper item | File |
|---|---|
| Table "Maritime transfer" (pooled 60-voyage mean/σ) | `experiments/final_rerun/results/main_reconstruction_raw.json` (TrajRAG) + `experiments/final_rerun/baseline_rescore/aggregate_*.json` (other baselines) |
| Table "Item4-matched" (short-horizon TrajRAG/AIS-ACNet) | `experiments/final_rerun/results/item4_matched_raw.json`, `experiments/final_rerun/ais_acnet_corrected/item4_points_*.json` |
| Table "Extended comparison" (8-method Houston table) | same files, aggregated by `experiments/final_rerun/student_followup/pairing.py` |
| Table "Distribution" / "Normalized error" | computed directly from `main_reconstruction_raw.json` + baseline details |
| Table "Statistical significance" (Wilcoxon + Holm) | `experiments/final_rerun/final_significance_all.json` (original Houston/LA tests) + `experiments/final_rerun/student_followup/results/task2_task3_full_stats.json` (completed LA/short-horizon tests, Holm correction, all 4 comparison families) |
| Figure "mean-difference forest plot" | `experiments/final_rerun/student_followup/figures/forest_plot_houston_main.png`, generated by `make_forest_plot.py` |
| Table "Corridor generalisation" | same `aggregate_*.json`/`main_reconstruction_raw.json` files, split by corridor |
| Table "Calibration" | `experiments/final_rerun/calibration_corrected.json` |
| Table "Maritime ablations" | `experiments/final_rerun/results/ablations_raw.json`, Holm-corrected in `student_followup/results/task2_task3_full_stats.json` (`F4_ablations`) |
| Table "Backbone sensitivity" | `experiments/r1_speedfix/results/` (predates the final correction, see paper caveat) |
| Table "Latency / cost" | `elapsed_sec`/`num_tokens` fields already inside `main_reconstruction_raw.json` |
| Table "Two simple kinematic baselines" | `experiments/final_rerun/student_followup/results/task4_task5_summary.json` (full per-voyage/per-step predictions in the sibling `task4_task5_main.json` / `task4_task5_item4.json`) |
| Figures "qualitative case studies" | `experiments/final_rerun/student_followup/figures/case_houston_arrival.png`, `case_la_transit.png`, selection logic in `task6_case_plots.py` |

**Every raw-result JSON in this repo stores full per-step predicted and
ground-truth positions**, not just the aggregate mean error — e.g. a
`main_reconstruction_raw.json` row looks like:

```json
{
  "corridor": "HOU-SHIPCHANNEL", "voyage_id": "219221000_2025-05-21T15:59:52",
  "phase": "arrival", "mean_error_km": 1.3,
  "predicted_positions": [[29.3452, -94.7109, 175.0], ...],
  "ground_truth_positions": [[29.34251, -94.70075, 175.0], ...],
  "mean_similar_positions": [[29.3544, -94.7528, 175.0], ...]
}
```

(`[lat, lon, minutes_since_phase_start]` per step.) This means every number
in the paper can be recomputed, re-tested, or re-plotted directly from these
files without ever calling the LLM again — which is exactly what
`experiments/final_rerun/student_followup/` does (see its module docstrings
for the exact supervisor-review task list it answers: full-precision
recomputation, paired bootstrap confidence intervals, completed/Holm-
corrected significance tests, two new training-free baselines, and
qualitative case plots).

## File and function reference

### `experiments/final_rerun/` (the authoritative, corrected pipeline)

- `voyage_manifest.json` — every voyage-selection list used anywhere in the
  paper (`seed=42`, deterministic): memory pool, usable test voyages, main
  reconstruction subset, ablation subset, item4-matched subset.
- `build_restricted_memory.py` — rebuilds summaries + FAISS embeddings for
  only the memory-pool voyages (read-only imports `generate_summary`/
  `get_embeddings` from `MARITIME-SPOOFING/data_pipeline/build_embeddings.py`).
- `final_rerun_engine.py` — `FinalRerunEngine`, see [§"The method"](#the-method-trajrag-engine).
- `run_main_reconstruction.py` / `run_ablations.py` / `run_item4_matched.py`
  — the three paid-execution scripts; checkpointed, print running cost.
- `calibration_corrected.py` — re-derives the kinematic guard's empirical
  displacement-ratio calibration from the corrected training data (free, no
  LLM calls — pure arithmetic over real AIS step-to-step displacements).
- `baseline_rescore/rescore_baselines.py` — rescores Kalman/ARIMA/LSTM/
  BiLSTM/Numeric-kNN on the exact corrected-data voyage subset TrajRAG used,
  saving full per-step points (read-only imports from `MARITIME-SPOOFING/
  methods/` and `baselines/knn_maritime/`).
- `ais_acnet_corrected/` — AIS-ACNet's per-voyage and per-step results after
  retraining on the corrected data (`per_voyage_*.json`: full-125 test set;
  `item4_points_*.json`: short-horizon matched subset).
- `student_followup/` — the most recent round of analysis (2026-11-01
  supervisor task list): precision recomputation, paired bootstrap CIs,
  completed/Holm-corrected Wilcoxon tests, the two new simple baselines, and
  the qualitative case-study plots. See `common.py`/`pairing.py` for the
  shared data-loading and haversine/destination-point utilities every other
  script in that folder builds on.

### `experiments/r1_speedfix/`

- `patched_engine.py` — `SpeedFixedRAGEngine(RAGEngine)`: the deployable
  speed-rule fix, retrieval-exclusion fix, and the three ablation modes
  (`no_retrieval`/`no_guard`/`no_heading`). Every later engine subclasses
  this one.

### `experiments/preprocessing_fix/`

- `corrected_build_voyages.py` — rebuilds `voyage_data_corrected_<CORRIDOR>.json`
  with causal (forward-fill) lat/lon resampling instead of the original
  bidirectional interpolation.

### `baselines/ais_acnet/`

- `build_voyage_windows*.py` — turns real, phase-labeled voyages into
  AIS-ACNet's windowed train/val/test tensors (voyage-anchored, not the
  original paper's arbitrary day-windows; see code comments for the
  `ip_mask` output-gating bug found and fixed while porting it here).
- `train_voyage_model*.py` — trains one network per phase per corridor.
- `train_corrected_all.py` — the final retraining run on the corrected data
  (150 epochs, patience 20, matching the documented methodology).

### `baselines/tslib_maritime/`

- `dataset.py` — builds the small, fixed, exact-fit window dataset per phase
  (no padding — a voyage is only included if its real half-length exactly
  matches the window).
- `model_wrapper.py` — builds any of the 7 supported architectures from a
  separately-cloned TSLib checkout (`TSLIB_PATH` env var, see
  [Provenance](#provenance-what-is-and-isnt-in-this-repo)).
- `train_tslib_model.py` / `run_all.py` — trains/evaluates one model × one
  phase, or all 7 × 3.

### `baselines/simple_baselines/` and `student_followup/task4_task5_baselines.py`

- `constant_position_predict()` — holds the last trusted position fixed.
- `heading_dr_predict()` — dead reckoning from the last trusted position,
  fixed last-trusted SOG, live trusted heading per step, propagated via
  `common.destination_point()` (spherical forward geodetic problem, same
  Earth radius and formula as the engine's own kinematic guard).

## Provenance: what is and isn't in this repo

This repo mirrors only the `Engineering Applications of Artificial
Intelligence/` folder of the authors' working tree, plus a **pruned,
read-only** copy of the handful of pre-existing files that folder's own
scripts import from or read directly (`MARITIME-SPOOFING/` above — about a
dozen files, not the full original application). **None of these vendored
files were modified for this release**; every correction described above is
implemented as a new subclass or a new script elsewhere in the repo, never
as an edit to these files.

**Not included, by design:**
- The aviation codebase (a separate, earlier project; see the conference
  paper, `paper/sec_intro.tex`, `\citep{felendler2026trajrag}`).
- `external/AIS-ACNet` — the original authors' code
  (https://github.com/yuyolshin/AIS-ACNet, MIT License). `baselines/
  ais_acnet/` here is new code that builds the dataset and training loop
  around that model; clone the original repo separately if you need its
  model definition.
- `thuml/Time-Series-Library` (https://github.com/thuml/Time-Series-Library,
  MIT License) — clone it and set `TSLIB_PATH` to use `baselines/
  tslib_maritime/`.
- Raw, per-day intermediate AIS extraction artifacts (`day_windows/`,
  `raw_by_day/`, `resampled_by_day/`, several GB) and trained model
  checkpoints for the 7 TSLib forecasters (~500MB) — both fully regenerable
  from the scripts above; excluded to keep this repository a reasonable
  size. AIS-ACNet's own (much smaller, ~16MB) checkpoints are included.
- The full `MARITIME-SPOOFING/` FastAPI application (`main.py`, the
  interactive frontend) — this repo only needs the offline engine class, not
  the live API server.

## Citation

```bibtex
@article{sasson2026trajragmaritime,
  title   = {Retrieval-augmented large language models for training-free
             trajectory reconstruction under satellite navigation spoofing
             in aviation and maritime surveillance},
  author  = {Sasson, Ruben and Felendler, Yuval and Elovici, Yuval and Shabtai, Asaf},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026}
}
```
