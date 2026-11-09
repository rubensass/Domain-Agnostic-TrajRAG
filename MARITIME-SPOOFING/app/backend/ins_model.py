"""
Dead-reckoning / marine-INS drift uncertainty model -- the vessel analog of
GPS-SPOOFING-APP/backend/ins_model.py (read-only reference; this is a fresh
implementation, not an edit of that file).

Models position error accumulation after GPS spoofing onset using:
- Constant drift rate (NM/hr) for the selected dead-reckoning system type
- Schuler oscillation (84.4-min period). This term applies to any inertial
  platform referenced to the Earth's surface, not just aircraft -- a marine
  INS exhibits the same pendulum-like error oscillation. Vessels relying on
  simple dead reckoning (gyrocompass heading + log speed, no inertial
  platform) don't have a literal Schuler pendulum, but we keep the same decay
  envelope as a conservative, consistent approximation across both domains.

Trajectory points are [lat, lon, minutes_since_start] 3-element lists.
"""

import math

EARTH_R = 6_371_000  # meters
SCHULER_PERIOD_HR = 84.4 / 60  # ~1.4067 hours

DRIFT_PRESETS = {
    "marine_ins":       0.5,   # high-end integrated marine INS
    "dead_reckoning":   3.0,   # gyrocompass heading + log speed, no current correction
    "worst_case_drift": 5.0,   # DR with uncorrected current/leeway
}

SYSTEM_LABELS = {
    "marine_ins":       "Marine INS",
    "dead_reckoning":   "Dead Reckoning (compass + log)",
    "worst_case_drift": "Worst-Case Drift (uncorrected current)",
}


def get_drift_rate(system_type: str) -> float:
    return DRIFT_PRESETS.get(system_type, 0.5)


def get_system_label(system_type: str) -> str:
    return SYSTEM_LABELS.get(system_type, system_type)


def haversine(p1, p2) -> float:
    """Distance in meters between two [lat, lon, ...] points."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing(p1, p2) -> float:
    """Forward azimuth in degrees from p1 to p2."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_point(point, bearing_deg, distance_m) -> dict:
    """Return {lat, lon} displaced from point by distance_m at bearing_deg."""
    lat1 = math.radians(point[0])
    lon1 = math.radians(point[1])
    b = math.radians(bearing_deg)
    d = distance_m / EARTH_R
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return {"lat": math.degrees(lat2), "lon": math.degrees(lon2)}


def compute_ins_uncertainty_band(ground_truth: list, drift_rate_nm_per_hr: float = 0.5) -> dict:
    """
    Compute the dead-reckoning uncertainty band for the spoofed segment.

    ground_truth: [[lat, lon, minutes_since_start], ...]
                  index 0 is the spoofing onset (elapsed time = 0)

    Returns band left/right edges, per-point drift in meters, max drift, and duration.
    """
    left, right, drift_per_point = [], [], []
    n = len(ground_truth)

    for i in range(n):
        elapsed_hours = (ground_truth[i][2] - ground_truth[0][2]) / 60.0
        raw_drift_m = drift_rate_nm_per_hr * 1852.0 * elapsed_hours
        schuler_factor = 1.0 - math.cos(2.0 * math.pi * elapsed_hours / SCHULER_PERIOD_HR)
        drift_m = raw_drift_m * schuler_factor

        drift_per_point.append(round(drift_m, 1))

        center = {"lat": ground_truth[i][0], "lon": ground_truth[i][1]}

        if drift_m < 1.0 or n < 2:
            left.append(center)
            right.append(center)
            continue

        if i == 0:
            fwd = bearing(ground_truth[0], ground_truth[1])
        else:
            fwd = bearing(ground_truth[i - 1], ground_truth[i])

        left.append(offset_point(ground_truth[i], (fwd - 90) % 360, drift_m))
        right.append(offset_point(ground_truth[i], (fwd + 90) % 360, drift_m))

    spoofing_duration_hr = (ground_truth[-1][2] - ground_truth[0][2]) / 60.0 if n > 1 else 0.0

    return {
        "left": left,
        "right": right,
        "drift_per_point_m": drift_per_point,
        "max_drift_m": round(max(drift_per_point), 1) if drift_per_point else 0.0,
        "spoofing_duration_hr": round(spoofing_duration_hr, 3),
    }
