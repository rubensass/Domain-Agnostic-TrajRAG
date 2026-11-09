"""
R1 (action plan, P0-blocking): the maritime kinematic validator currently
updates its reference speed using the GROUND-TRUTH SOG read from the withheld
AIS stream at each reconstructed step (MARITIME-SPOOFING/app/backend/rag_engine.py
line 471: `last_sog = point.get("sog") or last_sog`), rather than from an
LLM-inferred value. SOG is GNSS-derived and the paper's own domain-adapter
table marks it "untrusted after onset" -- so this is a real information leak
that may inflate TrajRAG's apparent accuracy, especially in phases where
speed changes fast (departure/arrival, per Section 6.7's own calibration
finding).

This module subclasses the ORIGINAL RAGEngine (imported read-only, never
edited) and overrides ONLY the speed-update line, via a full copy of
_predict_voyage with the single line changed -- everything else (retrieval,
LLM calls, kinematic validator math, output schema) is identical to the
original, byte-for-byte, so differences in results are attributable only to
the speed rule.

Two deployable variants, matching the action plan's R1a/R1b:
  "aviation_inferred" (R1a): v_i = haversine(pred_{i-1}, pred_i) / dt,
      the speed implied by the last two ACCEPTED reconstructed positions
      (falls back to the last trusted SOG for the first step) -- mirrors
      the aviation engine's own convention exactly.
  "last_trusted_held" (R1b): v_i = v_t for the whole blackout -- the
      last verified (pre-blackout) SOG, held constant. Simplest, and also
      fully deployable without any inference step.
  "oracle" (original behavior, kept only as a reference/ablation row,
      never as a "deployable" result): the current ground-truth-SOG rule.
"""
import json
import math
import sys
import time
from pathlib import Path

MARITIME_BACKEND = Path(__file__).resolve().parents[3] / "MARITIME-SPOOFING" / "app" / "backend"
if str(MARITIME_BACKEND) not in sys.path:
    sys.path.insert(0, str(MARITIME_BACKEND))

from rag_engine import RAGEngine, RESAMPLE_MINUTES, KINEMATIC_TOLERANCE  # noqa: E402

VALID_SPEED_RULES = {"oracle", "aviation_inferred", "last_trusted_held"}
VALID_ABLATIONS = {None, "no_retrieval", "no_guard", "no_heading"}

# ---------------------------------------------------------------------------
# R5-fix (supervisor review, item 5a): the ORIGINAL _find_similar_voyages
# (rag_engine.py:286-299) excludes only the exact query voyage from its
# corpus-wide FAISS index. It does NOT exclude the other 124 test voyages of
# the same corridor, so a test voyage can -- and does -- retrieve another
# test voyage as a "similar precedent" at evaluation time. This is test-set
# contamination of the retrieval memory, not a mere realism caveat, so it is
# fixed in code here rather than only disclosed in the manuscript. Loaded
# once at import time; corridor -> set of that corridor's test voyage ids.
# ---------------------------------------------------------------------------
_TEST_SELECTION_PATH = (
    Path(__file__).resolve().parents[3] / "MARITIME-SPOOFING" / "evaluation" / "results" / "test_voyage_selection.json"
)
with open(_TEST_SELECTION_PATH, "r", encoding="utf-8") as _f:
    _test_selection = json.load(_f)
TEST_IDS_BY_CORRIDOR = {
    corridor: {v["voyage_id"] for v in voyages} for corridor, voyages in _test_selection.items()
}


class SpeedFixedRAGEngine(RAGEngine):
    def _apply_kinematic_validation_alpha(
        self, predicted_lat, predicted_lon, last_lat, last_lon, last_sog_knots, alpha,
    ):
        """Same formula as the original _apply_kinematic_validation, but with
        alpha passed explicitly per call (R6) rather than read from the
        module-level KINEMATIC_TOLERANCE constant -- a global monkey-patch
        would race across the concurrently-running phases/threads this
        project uses, since departure and arrival need different alpha
        values at the same time."""
        dt_hours = RESAMPLE_MINUTES / 60.0
        max_dist_km = max(last_sog_knots, 0.0) * 1.852 * dt_hours * alpha
        max_dist_km = max(max_dist_km, 0.5)
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

    def _find_similar_voyages_no_test_leak(self, cd, partial_summary, target_id, phase, k):
        """R5-fix: same as the original _find_similar_voyages, but also
        excludes every voyage id in this corridor's test set (not only the
        exact target). This is now the DEFAULT retrieval path used by
        _predict_voyage below, replacing the original's self-exclusion-only
        behavior. Search depth widens dynamically (rather than a fixed
        overfetch) so it stays correct even if more than k+150 of the
        nearest neighbors happen to be excluded test voyages."""
        excluded = TEST_IDS_BY_CORRIDOR.get(cd.corridor_name, set()) | {target_id}
        q_vec = self._get_embeddings([partial_summary])
        index = cd.index_dictionary[phase]
        ids_for_phase = cd.ids_dictionary[phase]
        n_total = len(ids_for_phase)
        fetch_n = min(k + 150, n_total)
        while True:
            sims, idxs = index.search(q_vec, fetch_n)
            sims, idxs = sims[0], idxs[0]
            neighbors = [
                (ids_for_phase[i], float(sims[j]))
                for j, i in enumerate(idxs)
                if i != -1 and ids_for_phase[i] not in excluded
            ]
            if len(neighbors) >= k or fetch_n >= n_total:
                return neighbors[:k]
            fetch_n = min(fetch_n * 2, n_total)

    def _find_similar_voyages_vessel_disjoint(self, cd, partial_summary, target_id, phase, k):
        """R7: same as the original _find_similar_voyages, but also excludes
        any neighbor from the SAME VESSEL (same MMSI prefix of the voyage
        id) as the target, not only the exact same voyage -- the original
        only excludes self-matches, so a same-vessel historical trip
        (common: 93% of Houston test voyages have one in the training
        memory, since vessels make repeat trips through the channel over
        the 44-day window) could otherwise leak vessel-specific identity
        into retrieval. Overfetches (k+20 instead of k+1) since many
        same-vessel trips can cluster near the top of a similarity search."""
        target_mmsi = target_id.split("_")[0]
        q_vec = self._get_embeddings([partial_summary])
        index = cd.index_dictionary[phase]
        ids_for_phase = cd.ids_dictionary[phase]
        sims, idxs = index.search(q_vec, k + 20)
        sims, idxs = sims[0], idxs[0]
        neighbors = [
            (ids_for_phase[i], float(sims[j]))
            for j, i in enumerate(idxs)
            if ids_for_phase[i] != target_id and ids_for_phase[i].split("_")[0] != target_mmsi
        ]
        return neighbors[:k]

    def _predict_voyage(
        self, cd, voyage_id: str, phase: str, k: int, with_feedback_loop: bool,
        speed_rule: str = "last_trusted_held", ablation: str = None, alpha: float = None,
        vessel_disjoint: bool = False, max_steps: int = None,
    ) -> tuple:
        """
        R3 (action plan): `ablation` isolates one component at a time, using
        the same deployable speed rule as the main results throughout
        (never combined with the "oracle" rule):
          None            -- full pipeline (same as the main results)
          "no_retrieval"  -- the LLM gets the prefix, trusted heading and
                             environment, but no retrieved segments (tests
                             the value of the index itself)
          "no_guard"      -- every LLM proposal is accepted unclamped (tests
                             whether the guard drives accuracy or acts as a
                             safety net, mirroring the aviation ablation)
          "no_heading"    -- the live trusted gyrocompass heading reading is
                             never appended to the prompt during the
                             blackout (tests TrajRAG's own sensitivity to
                             that one input; NOT a comparison to any
                             separate heading-based baseline)

        `max_steps`, if set, reconstructs only the first `max_steps` points
        of the withheld half instead of the full one (item 4: matching the
        short, exact-fit horizon the modern forecasters are scored on, for a
        true same-timestamps comparison).
        """
        if speed_rule not in VALID_SPEED_RULES:
            raise ValueError(f"speed_rule must be one of {VALID_SPEED_RULES}")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(f"ablation must be one of {VALID_ABLATIONS}")

        target_voyage = cd.data_by_id[voyage_id]
        phase_data = target_voyage[phase]
        vessel_type = target_voyage["vessel_metadata"].get("vessel_name") or "commercial vessel"

        N = len(phase_data)
        spoof_index = N // 2
        observed_data = phase_data[:spoof_index]
        spoofed_data = phase_data[spoof_index:]
        if max_steps is not None:
            spoofed_data = spoofed_data[:max_steps]

        t_start = time.time()
        partial_summary, summary_tokens = self._generate_summary(phase, observed_data, vessel_type)
        total_tokens = summary_tokens

        last_verified = observed_data[-1] if observed_data else spoofed_data[0]
        last_lat, last_lon = last_verified["latitude"], last_verified["longitude"]
        last_trusted_sog = last_verified.get("sog") or 0.0
        last_sog = last_trusted_sog  # current reference speed for the validator

        predicted_positions: list = []
        ground_truth_positions: list = []
        mean_similar_positions: list = []
        last_token_count = 0

        for point in spoofed_data:
            t_min = point["minutes_since_start"]
            heading = point.get("heading") if ablation != "no_heading" else None

            if heading is not None:
                heading_prompt = (
                    f"{t_min:.1f} minutes into the {phase} phase, "
                    f"the vessel's gyrocompass heading read {heading} degrees."
                )
                partial_summary = f"{partial_summary} {heading_prompt}"

            if ablation == "no_retrieval":
                similar_ids, similar_summaries = [], []
            else:
                find_fn = (self._find_similar_voyages_vessel_disjoint if vessel_disjoint
                          else self._find_similar_voyages_no_test_leak)
                neighbors = find_fn(cd, partial_summary, voyage_id, phase, k)
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
                prev_lat, prev_lon = last_lat, last_lon
                if ablation == "no_guard":
                    pred_lat, pred_lon = float(prediction["lat"]), float(prediction["lon"])
                elif alpha is not None:
                    pred_lat, pred_lon = self._apply_kinematic_validation_alpha(
                        float(prediction["lat"]), float(prediction["lon"]),
                        last_lat, last_lon, last_sog, alpha,
                    )
                else:
                    pred_lat, pred_lon = self._apply_kinematic_validation(
                        float(prediction["lat"]), float(prediction["lon"]),
                        last_lat, last_lon, last_sog,
                    )
                predicted_positions.append([pred_lat, pred_lon, t_min])
                last_lat, last_lon = pred_lat, pred_lon

                # --- THE ONLY SUBSTANTIVE CHANGE vs. the original engine ---
                if speed_rule == "oracle":
                    last_sog = point.get("sog") or last_sog  # original (leaky) behavior
                elif speed_rule == "last_trusted_held":
                    last_sog = last_trusted_sog  # R1b: never updated during blackout
                elif speed_rule == "aviation_inferred":
                    dt_hours = RESAMPLE_MINUTES / 60.0
                    dist_km = self._haversine_km(prev_lat, prev_lon, pred_lat, pred_lon)
                    last_sog = (dist_km / 1.852) / dt_hours  # R1a: implied speed, in knots
                # -------------------------------------------------------------

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

    def run_experiment(
        self, corridor: str, voyage_id: str, phase: str, k: int = 6,
        with_feedback_loop: bool = True, speed_rule: str = "last_trusted_held",
        ablation: str = None, alpha: float = None, vessel_disjoint: bool = False,
        max_steps: int = None,
    ) -> dict:
        self.initialize()
        cd = self._get_corridor(corridor)
        if voyage_id not in cd.data_by_id:
            raise ValueError(f"Voyage '{voyage_id}' not found in corridor '{corridor}'.")

        (
            observed_data, ground_truth, predicted, mean_error_km,
            mean_similar, mean_similar_error_km, num_tokens, elapsed_sec,
        ) = self._predict_voyage(cd, voyage_id, phase, k, with_feedback_loop,
                                 speed_rule=speed_rule, ablation=ablation, alpha=alpha,
                                 vessel_disjoint=vessel_disjoint, max_steps=max_steps)

        return {
            "corridor": corridor, "voyage_id": voyage_id, "phase": phase,
            "speed_rule": speed_rule, "ablation": ablation, "alpha": alpha,
            "vessel_disjoint": vessel_disjoint, "max_steps": max_steps,
            "mean_error_km": mean_error_km,
            "mean_similar_error_km": mean_similar_error_km,
            "num_tokens": num_tokens,
            "elapsed_sec": round(elapsed_sec, 2),
            # Full per-step data (not just the aggregate mean), each as
            # [lat, lon, minutes_since_start]: requested so individual-point
            # results are never only recoverable as a pre-averaged scalar.
            "predicted_positions": predicted,
            "ground_truth_positions": ground_truth,
            "mean_similar_positions": mean_similar,
        }


speedfixed_engine = SpeedFixedRAGEngine()
