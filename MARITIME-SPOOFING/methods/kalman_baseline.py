"""
Classical-filtering baseline for vessel trajectory reconstruction -- the
maritime analog of methods/KALMAN.ipynb (read-only reference; fresh
implementation, not an edit of that notebook).

Constant-velocity Kalman filter with ocean-current drift injected at each
predict step (in place of the aircraft baseline's wind drift). State is
[lat, lon, v_lat, v_lon] -- no altitude dimension, since vessels operate on
the sea surface. Fed the observed (pre-spoofing) half of a phase with
predict+update; then run open-loop (predict only, no update) across the
spoofed half, exactly mirroring the aircraft study's "classical filtering
transitions to open-loop propagation under blackout" behavior.
"""

import math

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sog_cog_to_latlon_velocity(sog_knots, cog_deg, lat):
    """SOG (knots) + COG (deg) -> (dlat, dlon) per second, in degrees."""
    R = 6371e3
    sog_ms = sog_knots * 0.514444
    cog = math.radians(cog_deg)
    v_north = sog_ms * math.cos(cog)
    v_east = sog_ms * math.sin(cog)
    dlat = (v_north / R) * (180 / math.pi)
    dlon = (v_east / (R * math.cos(math.radians(lat)))) * (180 / math.pi)
    return dlat, dlon


def current_to_latlon_velocity(current_kmh, current_dir_deg, lat):
    """Ocean current velocity (km/h) + direction (deg, TO which it flows) -> (dlat, dlon)/s."""
    R = 6371e3
    v_ms = (current_kmh or 0.0) / 3.6
    cdir = math.radians(current_dir_deg or 0.0)
    v_north = v_ms * math.cos(cdir)
    v_east = v_ms * math.sin(cdir)
    dlat = (v_north / R) * (180 / math.pi)
    dlon = (v_east / (R * math.cos(math.radians(lat)))) * (180 / math.pi)
    return dlat, dlon


class KalmanFilterCurrentCV:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1
        self.H[1, 1] = 1

        self.Q = np.eye(4) * 1e-4
        self.R = np.eye(2) * 1e-3

        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

    def initialize(self, lat, lon, sog, cog):
        v_lat, v_lon = sog_cog_to_latlon_velocity(sog, cog, lat)
        self.x = np.array([[lat], [lon], [v_lat], [v_lon]], dtype=float)
        self.P = np.eye(4)

    def predict(self, current_lat=0.0, current_lon=0.0):
        self.x = self.F @ self.x
        self.x[0, 0] += current_lat * self.dt
        self.x[1, 0] += current_lon * self.dt
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape((2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()


def kalman_predict_phase(seq: list[dict]):
    """
    seq = list of point dicts for one voyage phase (must include latitude,
    longitude, sog, cog, ocean_current_velocity, ocean_current_direction).
    Returns (predictions[:, :2], ground_truth[:, :2]) for the second half.
    """
    mat = [
        [
            p["latitude"], p["longitude"], p.get("sog") or 0.0, p.get("cog") or 0.0,
            p.get("ocean_current_velocity") or 0.0, p.get("ocean_current_direction") or 0.0,
        ]
        for p in seq
    ]
    mat = np.array(mat, dtype=float)

    T = len(mat)
    mid = T // 2
    obs, future = mat[:mid], mat[mid:]

    lat0, lon0, sog0, cog0, _, _ = obs[0]
    kf = KalmanFilterCurrentCV(dt=1.0)
    kf.initialize(lat0, lon0, sog0, cog0)

    for i in range(1, mid):
        lat, lon, sog, cog, cur_v, cur_d = obs[i]
        c_lat, c_lon = current_to_latlon_velocity(cur_v, cur_d, lat)
        kf.predict(c_lat, c_lon)
        kf.update(np.array([lat, lon]))

    preds = []
    for i in range(len(future)):
        lat, lon, sog, cog, cur_v, cur_d = future[i]
        c_lat, c_lon = current_to_latlon_velocity(cur_v, cur_d, lat)
        kf.predict(c_lat, c_lon)
        preds.append(kf.get_state()[:2])

    preds = np.array(preds)
    gt = future[:, :2]
    return preds, gt


def evaluate_phase(sequences: list[list[dict]]) -> list[float]:
    """Mean Haversine error (km) per voyage for a list of phase sequences."""
    errors = []
    for seq in sequences:
        if len(seq) < 4:
            continue
        try:
            preds, gt = kalman_predict_phase(seq)
        except Exception:
            continue
        err = np.mean([haversine_km(gt[i][0], gt[i][1], preds[i][0], preds[i][1]) for i in range(len(preds))])
        errors.append(float(err))
    return errors
