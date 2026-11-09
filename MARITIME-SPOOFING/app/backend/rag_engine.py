"""
TrajRAG engine for commercial vessels -- adaptation of
GPS-SPOOFING-APP/backend/rag_engine.py (read-only reference; this is a fresh
implementation for the maritime domain, not an edit of that file).

Domain mapping vs. the aircraft engine:
  - ADS-B fields (lat/lon/track/altitude_ft/gspeed) -> AIS fields
    (latitude/longitude/heading/cog/sog)
  - Barometric altitude (GNSS-independent trusted signal, fed to the LLM at
    every step during the blackout) -> gyrocompass True Heading / rate of
    turn. COG and SOG are GPS-derived and therefore untrusted during spoofing,
    exactly like ADS-B lat/lon; Heading is not.
  - Takeoff / cruising / landing -> departure / transit / arrival
  - Kinematic validation: max feasible Haversine distance from the last
    verified SOG, over the fixed resampling interval, with the same 1.2x
    tolerance factor used in the aircraft study (TrajRAG paper, Sec 3.6).
    Ships are far slower than aircraft, so the admissible radius per step is
    much smaller in absolute terms.
"""

import json
import math
import re
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import OpenAI

from config import CORRIDORS, EMBED_MODEL, LLM_MODEL, get_openai_key

# Curated voyage picker: the demo UI only lists a curated subset of vessels
# per corridor (app/backend/curated_voyages.json). This narrows what's shown
# in /api/voyages only -- the full corpus (data_by_id, FAISS memory) is
# untouched, so retrieval quality during reconstruction is unaffected.
_CURATED_VOYAGES_PATH = Path(__file__).parent / "curated_voyages.json"
_curated_voyages: dict[str, set] = {}
if _CURATED_VOYAGES_PATH.exists():
    with open(_CURATED_VOYAGES_PATH, "r", encoding="utf-8") as _f:
        _curated_voyages = {k: set(v) for k, v in json.load(_f).items()}


def _with_retry(fn, *args, attempts=5, **kwargs):
    """Retry an OpenAI call with exponential backoff. Under concurrent load
    (many worker threads each issuing their own sequential call chains),
    transient rate-limit/timeout errors are expected and should not abort an
    otherwise-successful reconstruction."""
    last_err = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(min(30, 2 ** attempt))
    raise last_err
from ins_model import compute_ins_uncertainty_band, get_drift_rate

PHASES = ["departure", "transit", "arrival"]
RESAMPLE_MINUTES = 5           # must match data_pipeline/build_voyages.py
KINEMATIC_TOLERANCE = 1.2      # same margin used in the aircraft study

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_PHASE_TEMPLATE = """
You are given a segment of AIS data corresponding to the {phase_name} phase of
a commercial vessel's voyage through a coastal corridor.
Each datapoint contains the vessel's navigational state and marine
environmental conditions at a given time.

Keys per datapoint:
- latitude, longitude, heading, cog, sog (knots), minutes_since_start
- wave_height (m), wave_direction, wave_period (s), swell_wave_height (m),
  ocean_current_velocity (km/h), ocean_current_direction,
  sea_surface_temperature (C), wind_speed_10m (km/h), wind_direction_10m

Write a continuous, expert-level narrative describing this phase of the
voyage chronologically. For each datapoint:
1. Reference time via minutes_since_start.
2. Describe position (lat/lon), motion (heading, course over ground, speed).
3. Include the marine environmental context (sea state, swell, current, wind).
4. Mention relevant geographic features based on coordinates.
5. Use smooth transitions between datapoints.
6. Start with the vessel type: {vessel_type}

Write as a natural, continuous, professional description -- not bullet points.

Voyage phase data:
{phase_data}

Return the summary without any meta-text.
"""

PREDICT_POSITION_PROMPT = """
A vessel entered a GPS spoofing zone. Its position is now unknown.
Your task is to predict the vessel's latitude and longitude.

Known trajectory (before spoofing) and gyrocompass heading updates (heading is
independent of GPS and remains trusted throughout the blackout):
{partial_summary}

Summaries of {number_similar_voyages} similar voyages for the same phase:
{sim_text}

{predictions_history}

Based on these, predict the geographic position of the target vessel {predict_minutes:.1f} minutes after the beginning of the phase.
Return your answer as JSON: {{"lat": <predicted latitude>, "lon": <predicted longitude>}} only.
"""


# ---------------------------------------------------------------------------
# Per-corridor data container
# ---------------------------------------------------------------------------

class CorridorData:
    """Holds voyage data, summaries, and FAISS indexes for one corridor. Loaded lazily."""

    def __init__(self, corridor_name: str, cfg: dict):
        self.corridor_name = corridor_name
        self.cfg = cfg
        self.voyages_data: list = []
        self.data_by_id: dict = {}
        self.summary_by_phase: dict = {}
        self.index_dictionary: dict = {}
        self.ids_dictionary: dict = {}
        self._ready = False

    def initialize(self):
        if self._ready:
            return
        with open(self.cfg["voyage_data"], "r", encoding="utf-8") as f:
            self.voyages_data = json.load(f)
        with open(self.cfg["summaries"], "r", encoding="utf-8") as f:
            summaries = json.load(f)

        self.data_by_id = {v["voyage_id"]: v for v in self.voyages_data}
        self.summary_by_phase = {
            s["voyage_id"]: {p: s.get(p) for p in PHASES}
            for s in summaries
        }

        emb_dir = self.cfg["embeddings"]
        for phase in PHASES:
            idx_path = emb_dir / f"faiss_index_{phase}.index"
            ids_path = emb_dir / f"ids_{phase}.json"
            if idx_path.exists() and ids_path.exists():
                self.index_dictionary[phase] = faiss.read_index(str(idx_path))
                with open(ids_path, "r") as f:
                    self.ids_dictionary[phase] = json.load(f)

        self._ready = True

    def voyage_list(self) -> list[dict]:
        allowed = _curated_voyages.get(self.corridor_name)
        result = []
        for v in self.voyages_data:
            if allowed is not None and v["voyage_id"] not in allowed:
                continue
            meta = v["vessel_metadata"]
            first_ts = None
            for phase in PHASES:
                pts = v.get(phase, [])
                if pts:
                    first_ts = pts[0].get("timestamp")
                    break
            result.append({
                "voyage_id":   v["voyage_id"],
                "mmsi":        meta["mmsi"],
                "vessel_name": meta.get("vessel_name", ""),
                "vessel_type": meta.get("vessel_type", ""),
                "datetime":    first_ts,
            })
        return result


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RAGEngine:

    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._corridors: dict[str, CorridorData] = {}
        self._client_ready = False

    def initialize(self):
        if self._client_ready:
            return
        self.client = OpenAI(api_key=get_openai_key())
        self._client_ready = True

    def _get_corridor(self, corridor: str) -> CorridorData:
        if corridor not in CORRIDORS:
            raise ValueError(f"Unknown corridor '{corridor}'. Valid: {list(CORRIDORS)}")
        if corridor not in self._corridors:
            cd = CorridorData(corridor, CORRIDORS[corridor])
            cd.initialize()
            self._corridors[corridor] = cd
        return self._corridors[corridor]

    def list_corridors(self) -> list[dict]:
        return [{"id": k, "label": v["label"]} for k, v in CORRIDORS.items()]

    def list_voyages(self, corridor: str) -> list[dict]:
        return self._get_corridor(corridor).voyage_list()

    def run_experiment(
        self,
        corridor: str,
        voyage_id: str,
        phase: str,
        k: int = 6,
        with_feedback_loop: bool = True,
        drift_system_type: str = "marine_ins",
    ) -> dict:
        self.initialize()
        cd = self._get_corridor(corridor)

        if voyage_id not in cd.data_by_id:
            raise ValueError(f"Voyage '{voyage_id}' not found in corridor '{corridor}'.")
        if phase not in PHASES:
            raise ValueError(f"Phase must be one of {PHASES}.")

        (
            observed_data, ground_truth, predicted, mean_error_km,
            mean_similar, mean_similar_error_km, num_tokens, elapsed_sec,
        ) = self._predict_voyage(cd, voyage_id, phase, k, with_feedback_loop)

        ins_band = compute_ins_uncertainty_band(ground_truth, get_drift_rate(drift_system_type))

        meta = cd.data_by_id[voyage_id]["vessel_metadata"]
        first_ts = None
        for p in PHASES:
            pts = cd.data_by_id[voyage_id].get(p, [])
            if pts:
                first_ts = pts[0].get("timestamp")
                break

        return {
            "corridor":     corridor,
            "voyage_id":    voyage_id,
            "phase":        phase,
            "metadata":     {**meta, "datetime": first_ts},
            "observed":     [[p["latitude"], p["longitude"], p["minutes_since_start"]] for p in observed_data],
            "ground_truth": ground_truth,
            "predicted":    predicted,
            "mean_similar": mean_similar,
            "mean_error_km":         mean_error_km,
            "mean_similar_error_km": mean_similar_error_km,
            "num_tokens":   num_tokens,
            "elapsed_sec":  round(elapsed_sec, 2),
            "k":            k,
            "ins_band":       ins_band,
            "drift_system_type": drift_system_type,
        }

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _get_embeddings(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = _with_retry(self.client.embeddings.create, model=EMBED_MODEL, input=batch)
            vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
        X = np.vstack(vecs)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def _generate_summary(self, phase_name: str, phase_data: list, vessel_type: str) -> tuple[str, int]:
        prompt = PROMPT_PHASE_TEMPLATE.format(
            phase_name=phase_name, phase_data=json.dumps(phase_data), vessel_type=vessel_type,
        )
        resp = _with_retry(
            self.client.chat.completions.create,
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3,
        )
        return resp.choices[0].message.content.strip(), resp.usage.total_tokens

    def _find_similar_voyages(
        self, cd: CorridorData, partial_summary: str, target_id: str, phase: str, k: int
    ) -> list[tuple[str, float]]:
        q_vec = self._get_embeddings([partial_summary])
        index = cd.index_dictionary[phase]
        ids_for_phase = cd.ids_dictionary[phase]
        sims, idxs = index.search(q_vec, k + 1)
        sims, idxs = sims[0], idxs[0]
        neighbors = [
            (ids_for_phase[i], float(sims[j]))
            for j, i in enumerate(idxs)
            if ids_for_phase[i] != target_id
        ]
        return neighbors[:k]

    @staticmethod
    def _clean_llm_json(raw: str) -> Optional[dict]:
        cleaned = re.sub(r"^```(?:json|python)?", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"```$", "", cleaned.strip())
        cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            cleaned2 = cleaned.replace("\\n", " ").replace('\\"', '"')
            return json.loads(cleaned2)

    def _predict_position_from_prompt(
        self, partial_summary: str, similar_summaries: list[str],
        predict_minutes: float, prediction_history: list,
    ) -> tuple[Optional[dict], int]:
        sim_text = "\n\n".join(
            [f"Similar voyage {i+1}:\n{s}" for i, s in enumerate(similar_summaries)]
        )
        if prediction_history:
            predictions_history = (
                "Here are the predictions you already made "
                "(list of [latitude, longitude, minutes_since_start]). "
                "Ensure the next prediction is consistent with realistic vessel dynamics "
                "and does not repeat or reverse position.\n\n"
                f"PREDICTION HISTORY: {prediction_history}"
            )
        else:
            predictions_history = ""

        prompt = PREDICT_POSITION_PROMPT.format(
            partial_summary=partial_summary,
            number_similar_voyages=len(similar_summaries),
            sim_text=sim_text,
            predict_minutes=predict_minutes,
            predictions_history=predictions_history,
        )
        resp = _with_retry(
            self.client.chat.completions.create,
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        result = self._clean_llm_json(resp.choices[0].message.content)
        return result, resp.usage.total_tokens

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def _apply_kinematic_validation(
        self, predicted_lat: float, predicted_lon: float,
        last_lat: float, last_lon: float, last_sog_knots: float,
    ) -> tuple[float, float]:
        """
        Phase-specific kinematic validation (TrajRAG paper Sec 3.6): the LLM's
        proposal must fall within the max feasible displacement implied by the
        last verified SOG over the fixed sampling interval, times a 1.2x
        tolerance margin. If not, project onto the boundary of the admissible
        motion envelope.
        """
        dt_hours = RESAMPLE_MINUTES / 60.0
        max_dist_km = max(last_sog_knots, 0.0) * 1.852 * dt_hours * KINEMATIC_TOLERANCE
        max_dist_km = max(max_dist_km, 0.5)  # floor: allow at least ~0.5km for near-zero SOG cases

        actual_dist_km = self._haversine_km(last_lat, last_lon, predicted_lat, predicted_lon)
        if actual_dist_km <= max_dist_km:
            return predicted_lat, predicted_lon

        bearing = self._bearing_deg(last_lat, last_lon, predicted_lat, predicted_lon)
        R = 6371.0
        d = max_dist_km / R
        lat1, lon1, b = map(math.radians, [last_lat, last_lon, bearing])
        lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
        lon2 = lon1 + math.atan2(
            math.sin(b) * math.sin(d) * math.cos(lat1),
            math.cos(d) - math.sin(lat1) * math.sin(lat2),
        )
        return math.degrees(lat2), math.degrees(lon2)

    @staticmethod
    def _get_mean_similar_position(
        similar_ids: list[str], data_by_id: dict, phase: str, t_min: float
    ) -> Optional[list]:
        lats, lons = [], []
        for vid in similar_ids:
            voyage = data_by_id.get(vid)
            if not voyage:
                continue
            phase_data = voyage.get(phase, [])
            if not phase_data:
                continue
            closest = min(phase_data, key=lambda p: abs(p["minutes_since_start"] - t_min))
            if closest.get("latitude") is not None and closest.get("longitude") is not None:
                lats.append(closest["latitude"])
                lons.append(closest["longitude"])
        if not lats:
            return None
        return [sum(lats) / len(lats), sum(lons) / len(lons), t_min]

    def _predict_voyage(
        self, cd: CorridorData, voyage_id: str, phase: str, k: int, with_feedback_loop: bool,
    ) -> tuple:
        target_voyage = cd.data_by_id[voyage_id]
        phase_data = target_voyage[phase]
        vessel_type = target_voyage["vessel_metadata"].get("vessel_name") or "commercial vessel"

        N = len(phase_data)
        spoof_index = N // 2
        observed_data = phase_data[:spoof_index]
        spoofed_data = phase_data[spoof_index:]

        t_start = time.time()
        partial_summary, summary_tokens = self._generate_summary(phase, observed_data, vessel_type)
        total_tokens = summary_tokens

        last_verified = observed_data[-1] if observed_data else spoofed_data[0]
        last_lat, last_lon = last_verified["latitude"], last_verified["longitude"]
        last_sog = last_verified.get("sog") or 0.0

        predicted_positions: list = []
        ground_truth_positions: list = []
        mean_similar_positions: list = []
        last_token_count = 0

        for point in spoofed_data:
            t_min = point["minutes_since_start"]
            heading = point.get("heading")

            # Heading is gyrocompass-derived and independent of GPS, so it
            # remains a trusted real-time update during the blackout -- the
            # vessel analog of the aircraft engine's barometric altitude feed.
            if heading is not None:
                heading_prompt = (
                    f"{t_min:.1f} minutes into the {phase} phase, "
                    f"the vessel's gyrocompass heading read {heading} degrees."
                )
                partial_summary = f"{partial_summary} {heading_prompt}"

            neighbors = self._find_similar_voyages(cd, partial_summary, voyage_id, phase, k)
            similar_ids = [vid for vid, _ in neighbors]
            similar_summaries = [
                cd.summary_by_phase[vid][phase]
                for vid in similar_ids
                if cd.summary_by_phase.get(vid, {}).get(phase)
            ]

            history = predicted_positions if with_feedback_loop else []
            prediction, last_token_count = self._predict_position_from_prompt(
                partial_summary, similar_summaries, t_min, history
            )
            total_tokens += last_token_count

            if prediction:
                pred_lat, pred_lon = self._apply_kinematic_validation(
                    float(prediction["lat"]), float(prediction["lon"]),
                    last_lat, last_lon, last_sog,
                )
                predicted_positions.append([pred_lat, pred_lon, t_min])
                last_lat, last_lon = pred_lat, pred_lon
                last_sog = point.get("sog") or last_sog

            ground_truth_positions.append([point["latitude"], point["longitude"], t_min])

            mean_pos = self._get_mean_similar_position(similar_ids, cd.data_by_id, phase, t_min)
            if mean_pos:
                mean_similar_positions.append(mean_pos)

        mean_error = 0.0
        for pred, gt in zip(predicted_positions, ground_truth_positions):
            mean_error += self._haversine_km(pred[0], pred[1], gt[0], gt[1])
        if predicted_positions:
            mean_error /= len(predicted_positions)

        mean_similar_error = 0.0
        for sim, gt in zip(mean_similar_positions, ground_truth_positions):
            mean_similar_error += self._haversine_km(sim[0], sim[1], gt[0], gt[1])
        if mean_similar_positions:
            mean_similar_error /= len(mean_similar_positions)

        elapsed_sec = time.time() - t_start
        return (
            observed_data, ground_truth_positions, predicted_positions,
            round(mean_error, 2), mean_similar_positions, round(mean_similar_error, 2),
            total_tokens, elapsed_sec,
        )


# Module-level singleton
engine = RAGEngine()
