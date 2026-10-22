"""Reduced, fast model of the SDL5 formulation cell's mixer-loading cycle.

A UR3e on the workstation deck opens the ARE-312 mixer's lid, takes the cup
from the balance inside the FormulatorCNC_V2, puts it in the mixer's holder
and closes the lid. The geometry (robot chain, link collision spheres, mixer
lid hinge, handle, bowl seat, CNC and balance boxes, deck) is exported from
the sdl_dt digital twin into data/cell_geometry.json by extract_geometry.py.
The equations are documented in PROBLEM.md; the comments below cite them as
(E1)...(E12).

`evaluate(x)` returns every term of the objective; `objective(x)` the scalar.
Pure numpy, deterministic, no state beyond the loaded geometry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GEO = json.loads((Path(__file__).resolve().parent / "data" / "cell_geometry.json").read_text())

# ------------------------------------------------------------------ decision vector
# x = [base_x, base_y, base_yaw, cnc_x, cnc_y, lid_open, tilt_x, tilt_y, pick_az,
#      shoulder_A, elbow_A, wrist_A, shoulder_B, elbow_B, wrist_B]
# Branch A: lid handling and the pick from the balance; branch B: the moves in and
# out of the mixer's holder (sdl_dt re-seeds IK there, so the arm may reconfigure).
DECK = GEO["deck"]
_FOOT = np.array(GEO["cnc"]["footprint"])  # CNC footprint, CNC frame [[xlo, ylo], [xhi, yhi]]
_R_BASE = GEO["ur3e"]["base_footprint_radius"]
NAMES = [
    "base_x", "base_y", "base_yaw", "cnc_x", "cnc_y", "lid_open",
    "tilt_x", "tilt_y", "pick_azimuth",
    "shoulder_A", "elbow_A", "wrist_A", "shoulder_B", "elbow_B", "wrist_B",
]
BOUNDS = np.array(
    [
        [DECK["x"][0] + _R_BASE, DECK["x"][1] - _R_BASE],
        [-0.70, DECK["y"][1] - _R_BASE],
        [-math.pi, math.pi],
        [DECK["x"][0] - _FOOT[0, 0], DECK["x"][1] - _FOOT[1, 0]],
        [DECK["y"][0] - _FOOT[0, 1], DECK["y"][1] - _FOOT[1, 1]],
        [math.radians(20.0), GEO["mixer"]["lid_open_max"]],
        [math.radians(-30.0), math.radians(30.0)],
        [math.radians(-30.0), math.radians(30.0)],
        [0.0, 2 * math.pi],
    ]
    + [[0.0, 1.0]] * 6
)

# ------------------------------------------------------------------ constants
V_JOINT, A_JOINT = 1.05, 1.4  # rad/s, rad/s^2: UR movej defaults (60 deg/s, 80 deg/s^2)
V_LIN, A_LIN = 0.25, 1.2  # m/s, m/s^2: UR movel defaults
V_HANDLE, A_HANDLE = 0.10, 0.5  # m/s, m/s^2: handle speed on the lid arc (assumed)
C_MIN = GEO["planner"]["min_clearance"]  # 5 mm, as plan_lid_cycle(min_clearance=0.005)
JUMP = GEO["planner"]["ik_continuity_rad"]  # 0.25 rad, as _Planner._solve
W_MIN = 0.01  # manipulability normalization used by plan_lid_cycle's score
# Tiered penalties (E12): unreachable layouts score >= W_INFEASIBLE; reachable
# ones score cycle time + a penalty saturating at P_SAT < W_INFEASIBLE - T.
W_INFEASIBLE, W_REACH = 1000.0, 1000.0  # s, s/m
W_COLLIDE, W_JUMP, W_MANIP = 500.0, 20.0, 2.0  # s/m, s/rad, s per unit shortfall
P_SAT = 900.0  # s
N_LINE, N_ARC, N_JOINT = 5, 7, 5  # samples per linear move, lid arc, joint move
READY_FALLBACKS = ((0.35, 0.10), (0.35, 0.05), (0.3, 0.0), (0.2, 0.0))  # (back, lift), as _plan_from

_DH = GEO["ur3e"]["dh"]
D1, D4, D5, D6 = _DH["d"][0], _DH["d"][3], _DH["d"][4], _DH["d"][5]
A2, A3 = _DH["a"][1], _DH["a"][2]
TCP = GEO["ur3e"]["tcp_offset"]
_LIM = np.array(GEO["ur3e"]["joint_limits"])
PLATE = GEO["ur3e"]["plate_height"]
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
RZ_PI = np.diag([-1.0, -1.0, 1.0, 1.0])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _trans(x, y, z):
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


# ------------------------------------------------------------------ robot (E1-E3)
_CHAIN = [(np.array(j["origin"]), j["kind"] != "fixed", j["child"]) for j in GEO["ur3e"]["chain"]]
_SPHERE_LINKS = list(GEO["ur3e"]["link_spheres"])
_SPHERES = [(np.array(GEO["ur3e"]["link_spheres"][l]["centres"]), np.array(GEO["ur3e"]["link_spheres"][l]["radii"])) for l in _SPHERE_LINKS]
_GR = GEO["gripper"]
# Gripper and held cup as spheres along the tool / cup z axis (frame-local centres, radii).
_GRIP_BODY = (np.array([[0, 0, 0.0], [0, 0, 0.04]]), np.full(2, _GR["body_radius"]))
_GRIP_FING = (np.array([[0, 0, 0.088], [0, 0, 0.11]]), np.full(2, 0.022))
_CUPG = GEO["cup"]
_CUP_SPH = (
    np.array([[0, 0, z] for z in np.linspace(0.012, _CUPG["height"] - 0.012, 3)]),
    np.full(3, math.hypot(_CUPG["radius"], 0.0115)),
)


def fk(base: np.ndarray, q: np.ndarray):
    """Batched forward kinematics (E1). q: (N, 6). Returns link frames
    {child: (N,4,4)}, joint z axes (N,6,3) and joint origins (N,6,3), all world."""
    n = len(q)
    t = np.broadcast_to(base, (n, 4, 4)).copy()
    frames, axes, origins = {}, [], []
    k = 0
    for origin, revolute, child in _CHAIN:
        t = t @ origin
        if revolute:
            axes.append(t[:, :3, 2].copy())
            origins.append(t[:, :3, 3].copy())
            c, s = np.cos(q[:, k]), np.sin(q[:, k])
            rot = np.zeros((n, 4, 4))
            rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1] = c, -s, s, c
            rot[:, 2, 2] = rot[:, 3, 3] = 1.0
            t = t @ rot
            k += 1
        frames[child] = t
    return frames, np.stack(axes, 1), np.stack(origins, 1)


def _dh_a(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), math.cos(alpha), math.sin(alpha)
    n = len(theta)
    m = np.zeros((n, 4, 4))
    m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 0, 3] = ct, -st * ca, st * sa, a * ct
    m[:, 1, 0], m[:, 1, 1], m[:, 1, 2], m[:, 1, 3] = st, ct * ca, -ct * sa, a * st
    m[:, 2, 1], m[:, 2, 2], m[:, 2, 3] = sa, ca, d
    m[:, 3, 3] = 1.0
    return m


def ik(base_inv: np.ndarray, tcp_targets: np.ndarray, shoulder: int, elbow: int, wrist: int):
    """Closed-form UR inverse kinematics on one branch (E2), batched over targets.

    tcp_targets: (M,4,4) TCP poses in the cell frame. shoulder, elbow, wrist: +1/-1.
    Returns q (M,6) and the reach deficit (M,) in metres (0 where reachable):
    how far the target lies outside the region each closed-form step can solve.
    """
    t06 = RZ_PI @ base_inv @ tcp_targets @ _trans(0, 0, -TCP)  # tool0 in the DH base frame
    p = t06[:, :3, 3]
    p05 = p - D6 * t06[:, :3, 2]
    r = np.hypot(p05[:, 0], p05[:, 1])
    deficit = np.maximum(D4 - r, 0.0)
    phi = np.arccos(np.clip(D4 / np.maximum(r, 1e-9), -1, 1))
    th1 = np.arctan2(p05[:, 1], p05[:, 0]) + shoulder * phi + math.pi / 2
    s1, c1 = np.sin(th1), np.cos(th1)
    arg5 = (p[:, 0] * s1 - p[:, 1] * c1 - D4) / D6
    deficit += D6 * np.maximum(np.abs(arg5) - 1.0, 0.0)
    th5 = wrist * np.arccos(np.clip(arg5, -1, 1))
    t60 = np.linalg.inv(t06)
    s5 = np.sin(th5)
    s5 = np.where(np.abs(s5) < 1e-9, 1e-9, s5)
    th6 = np.arctan2((-t60[:, 1, 0] * s1 + t60[:, 1, 1] * c1) / s5, (t60[:, 0, 0] * s1 - t60[:, 0, 1] * c1) / s5)
    t01 = _dh_a(th1, D1, 0.0, _DH["alpha"][0])
    t45 = _dh_a(th5, D5, 0.0, _DH["alpha"][4])
    t56 = _dh_a(th6, D6, 0.0, _DH["alpha"][5])
    t14 = np.linalg.inv(t01) @ t06 @ np.linalg.inv(t45 @ t56)
    p14x, p14y = t14[:, 0, 3], t14[:, 1, 3]  # joints 2-4 are parallel: a planar 2-link arm in frame 1's x-y plane
    l = np.hypot(p14x, p14y)
    deficit += np.maximum(l - (abs(A2) + abs(A3)), 0.0) + np.maximum(abs(abs(A2) - abs(A3)) - l, 0.0)
    c3 = np.clip((l**2 - A2**2 - A3**2) / (2 * A2 * A3), -1, 1)
    th3 = elbow * np.arccos(c3)
    th2 = np.arctan2(p14y, p14x) - np.arctan2(A3 * np.sin(th3), A2 + A3 * np.cos(th3))
    t12 = _dh_a(th2, 0.0, A2, 0.0)
    t23 = _dh_a(th3, 0.0, A3, 0.0)
    t34 = np.linalg.inv(t12 @ t23) @ t14
    th4 = np.arctan2(t34[:, 1, 0], t34[:, 0, 0])
    return np.stack([th1, th2, th3, th4, th5, th6], 1), deficit


def unwrap(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Sequentially choose each configuration's 2*pi turns nearest the previous
    one (E3), starting from `ref`, within the joint limits (+-2*pi; the elbow,
    limited to +-pi, is never turned). Every choice is the same pose."""
    out = q.copy()
    prev = ref
    for i in range(len(q)):
        cand = q[i][None, :] + 2 * math.pi * np.arange(-2, 3)[:, None]  # (5, 6)
        ok = (cand >= _LIM[:, 0] - 1e-9) & (cand <= _LIM[:, 1] + 1e-9)
        cost = np.where(ok, np.abs(cand - prev), np.inf)
        out[i] = cand[np.argmin(cost, axis=0), np.arange(6)]
        prev = out[i]
    return out


# ------------------------------------------------------------------ cell targets (E4-E6)
_MX = GEO["mixer"]
_CUP_SEAT = np.array(_MX["cup_seat"])
_CUP_ON_SCALE = np.array(GEO["cnc"]["cup_on_scale"])
_PL = GEO["planner"]


def lid_frame(a: float) -> np.ndarray:
    """thinky_cover frame at lid angle a: hinge at z = 0.328, axis -y (E4)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, _MX["lid_hinge_z"]], [0, 0, 0, 1.0]])


def handle_grasp(a: float, tilt) -> np.ndarray:
    """TCP holding the lid handle at lid angle a, tilted by (tilt_x, tilt_y) (E4)."""
    return lid_frame(a) @ _trans(*_MX["handle_in_lid"]) @ FLIP @ _rx(tilt[0]) @ _ry(tilt[1])


def cup_grasp(azimuth: float) -> np.ndarray:
    """TCP in the cup frame, pinching the rim wall at `azimuth` (E5)."""
    r = _CUPG["radius"] - _CUPG["wall"] / 2
    place = _trans(r * math.cos(azimuth), r * math.sin(azimuth), _CUPG["height"] - _CUPG["grasp_depth"]) @ _rz(azimuth)
    return place @ FLIP @ _rx(_CUPG["grasp_tilt"])


def back_off(t: np.ndarray, d: float) -> np.ndarray:
    out = t.copy()
    out[:3, 3] -= d * t[:3, 2]
    return out


def along(t: np.ndarray, d: float) -> np.ndarray:
    out = t.copy()
    out[:3, 3] += d * t[:3, 2]
    return out


def lifted(t: np.ndarray, dz: float) -> np.ndarray:
    out = t.copy()
    out[2, 3] += dz
    return out


def _log_rot(r: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if math.pi - angle < 1e-6:  # half turn: axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(r) + 1) / 2, 0))
        axis *= np.sign(w + 1e-12)
        return axis * angle
    return w / (2 * math.sin(angle)) * angle


def _exp_rot(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def line_poses(a: np.ndarray, b: np.ndarray, n: int) -> tuple[np.ndarray, float, float]:
    """n poses on the straight line a -> b with single-axis rotation (E6);
    also the path length (m) and rotation angle (rad)."""
    v = _log_rot(b[:3, :3] @ a[:3, :3].T)
    out = np.empty((n, 4, 4))
    for i, s in enumerate(np.linspace(0.0, 1.0, n)):
        out[i] = np.eye(4)
        out[i, :3, :3] = _exp_rot(s * v) @ a[:3, :3]
        out[i, :3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]
    return out, float(np.linalg.norm(b[:3, 3] - a[:3, 3])), float(np.linalg.norm(v))


# ------------------------------------------------------------------ timing (E7)
def trap(dist, v, a):
    """Trapezoidal (or triangular) profile time for distance `dist` (E7)."""
    dist = np.abs(dist)
    return np.where(dist < v * v / a, 2 * np.sqrt(dist / a), dist / v + v / a)


def joint_time(q0, q1) -> float:
    return float(trap(q1 - q0, V_JOINT, A_JOINT).max())


# ------------------------------------------------------------------ collision (E9)
_BODY_LO, _BODY_HI = np.array(_MX["body_box"][0]), np.array(_MX["body_box"][1])
_LID_LO, _LID_HI = np.array(_MX["lid_box"][0]), np.array(_MX["lid_box"][1])
_CAV = _MX["cavity"]
_CNC_BOXES = np.array(GEO["cnc"]["boxes"])  # (B, 2, 3) CNC frame, axis aligned (CNC yaw fixed at 0)
_CNC_ALL = np.array([_CNC_BOXES[:, 0].min(0), _CNC_BOXES[:, 1].max(0)])


def _box_sdf(p: np.ndarray, lo, hi) -> np.ndarray:
    """Signed distance from points (..., 3) to an axis-aligned box (negative inside)."""
    outside = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=-1)
    inside = np.minimum(p - lo, hi - p).min(axis=-1)
    return np.where(outside > 0, outside, -inside)


def _mixer_sdf(p: np.ndarray) -> np.ndarray:
    """Mixer body box, open over the bowl cavity (as formulation._mixer_body_distance)."""
    d = _box_sdf(p, _BODY_LO, _BODY_HI)
    cx, cy, rad, floor = _CAV
    r = np.hypot(p[..., 0] - cx, p[..., 1] - cy)
    cav = (r < rad) & (p[..., 2] > floor) & (d < 0)
    return np.where(cav, np.minimum(rad - r, p[..., 2] - floor), d)


def _place(frames: np.ndarray, spheres) -> np.ndarray:
    centres, _ = spheres
    return np.einsum("nij,sj->nsi", frames[:, :3, :3], centres) + frames[:, None, :3, 3]


# Sphere groups: 0 = robot base housing (no floor check), 1 = moving arm links,
# 2 = gripper, 3 = held cup.
_GROUP = np.concatenate(
    [np.full(len(r), 0 if l == "base_link_inertia" else 1) for l, (_, r) in zip(_SPHERE_LINKS, _SPHERES)]
    + [np.full(len(_GRIP_BODY[1]) + len(_GRIP_FING[1]), 2), np.full(len(_CUP_SPH[1]), 3)]
)
_RADII = np.concatenate([r for _, r in _SPHERES] + [_GRIP_BODY[1], _GRIP_FING[1], _CUP_SPH[1]])


def clearances(base, q, lid, contact, cup_frames, cnc_xy) -> np.ndarray:
    """Smallest signed sphere-surface distance of arm, gripper and held cup to the
    mixer body, the lid, the CNC and balances, and the floor, per configuration (E9).
    In contact phases the gripper and cup are left out of the mixer/lid/CNC
    checks (they are meant to touch the handle or the cup), as in sdl_dt."""
    frames, _, _ = fk(base, q)
    held = ~np.isnan(cup_frames[:, 0, 0])
    cf = np.where(held[:, None, None], cup_frames, np.eye(4))
    pts = np.concatenate(
        [_place(frames[l], sph) for l, sph in zip(_SPHERE_LINKS, _SPHERES)]
        + [_place(frames["tool0"], _GRIP_BODY), _place(frames["tool0"], _GRIP_FING), _place(cf, _CUP_SPH)],
        axis=1,
    )  # (N, S, 3)
    d = _mixer_sdf(pts)
    c, s = np.cos(lid), np.sin(lid)  # lid frame (E4): local = R^T (p - t), R = rot(-y, a)
    rel = pts - np.array([0.0, 0.0, _MX["lid_hinge_z"]])
    local = np.stack([c[:, None] * rel[..., 0] + s[:, None] * rel[..., 2], rel[..., 1],
                      -s[:, None] * rel[..., 0] + c[:, None] * rel[..., 2]], -1)
    d = np.minimum(d, _box_sdf(local, _LID_LO, _LID_HI))
    local = pts - np.array([cnc_xy[0], cnc_xy[1], 0.0])
    cnc = _box_sdf(local, _CNC_ALL[0], _CNC_ALL[1])
    near = cnc - _RADII < 0.05
    if near.any():
        lp = local[near][:, None, :]
        per_box = _box_sdf(lp, _CNC_BOXES[None, :, 0], _CNC_BOXES[None, :, 1]).min(1)
        cnc[near] = per_box
    d = np.minimum(d, cnc) - _RADII
    excluded = (contact[:, None] & (_GROUP >= 2)[None]) | (~held[:, None] & (_GROUP == 3)[None])
    d = np.where(excluded, np.inf, d)
    floor = np.where((_GROUP == 0)[None] | (~held[:, None] & (_GROUP == 3)[None]), np.inf, pts[..., 2] - _RADII)
    return np.minimum(d, floor).min(1)


def manipulability(base, q) -> np.ndarray:
    """Yoshikawa measure sqrt(det(J J^T)) at the TCP (E10), batched."""
    frames, axes, origins = fk(base, q)
    tcp = (frames["tool0"] @ _trans(0, 0, TCP))[:, :3, 3]
    jv = np.cross(axes, tcp[:, None, :] - origins)
    j = np.concatenate([jv, axes], 2).transpose(0, 2, 1)  # (N,6,6)
    return np.sqrt(np.maximum(np.linalg.det(j @ j.transpose(0, 2, 1)), 0.0))


# ------------------------------------------------------------------ footprints (E11)
_MIXER_RECT = np.array([[_BODY_LO[0], _BODY_LO[1]], [_BODY_HI[0], _BODY_HI[1]]])


def _rect_overlap(a, b) -> float:
    ox = min(a[1, 0], b[1, 0]) - max(a[0, 0], b[0, 0])
    oy = min(a[1, 1], b[1, 1]) - max(a[0, 1], b[0, 1])
    return max(0.0, min(ox, oy)) if ox > 0 and oy > 0 else 0.0


def _disk_rect_overlap(c, r, rect) -> float:
    q = np.clip(c, rect[0], rect[1])
    d = float(np.linalg.norm(c - q))
    if d > 0:
        return max(0.0, r - d)
    inside = float(min(c[0] - rect[0, 0], rect[1, 0] - c[0], c[1] - rect[0, 1], rect[1, 1] - c[1]))
    return r + inside


def footprint_overlap(base_xy, cnc_xy) -> float:
    cnc_rect = _FOOT + cnc_xy
    return (
        _disk_rect_overlap(base_xy, _R_BASE, _MIXER_RECT)
        + _disk_rect_overlap(base_xy, _R_BASE, cnc_rect)
        + _rect_overlap(cnc_rect, _MIXER_RECT)
    )


# ------------------------------------------------------------------ the cycle (E8, E12)
def evaluate(x) -> dict:
    """All terms of the objective at decision vector x (length 15)."""
    x = np.asarray(x, float)
    bx, by, yaw, cx, cy, lid_open, tx, ty, az = x[:9]
    branch_a = tuple(1 if f >= 0.5 else -1 for f in x[9:12])
    branch_b = tuple(1 if f >= 0.5 else -1 for f in x[12:15])
    base = _trans(bx, by, PLATE) @ _rz(yaw)
    base_inv = np.linalg.inv(base)
    cnc_xy = np.array([cx, cy])
    tilt = (tx, ty)
    out = {"footprint_overlap": footprint_overlap(np.array([bx, by]), cnc_xy)}

    # Key TCP targets (E5, E8).
    g_closed, g_open = handle_grasp(0.0, tilt), handle_grasp(lid_open, tilt)
    a_closed, a_open = back_off(g_closed, _PL["standoff"]), back_off(g_open, _PL["standoff"])
    # Ready pose: drawn back toward the arm and lifted; the first of sdl_dt's
    # fallbacks (_plan_from) that branch A reaches, else the nearest miss.
    toward = np.array([bx, by, 0.0]) - a_closed[:3, 3]
    toward[2] = 0.0
    readies = []
    for back, lift in READY_FALLBACKS:
        r_ = a_closed.copy()
        r_[:3, 3] += back * toward + np.array([0, 0, lift])
        readies.append(r_)
    ready_def = ik(base_inv, np.stack(readies), *branch_a)[1]
    ok = np.flatnonzero(ready_def == 0)
    ready = readies[ok[0] if len(ok) else int(np.argmin(ready_def))]
    cup0 = _trans(cx, cy, 0.0) @ _CUP_ON_SCALE
    pick_g = cup_grasp(az)
    pick = cup0 @ pick_g
    pick_up = lifted(pick, _CUPG["lift"])
    released = along(_CUP_SEAT, _CUPG["release_height"])
    place_g = cup_grasp(_CUPG["place_azimuth"])
    place = released @ place_g
    pre = along(released, _CUPG["standoff"]) @ place_g
    over = lifted(pre, _PL["over_rise"])
    high = over.copy()
    high[:3, 3] = (pick_up[:3, 3] + over[:3, 3]) / 2
    high[2, 3] = max(pick_up[2, 3], over[2, 3]) + _PL["high_rise"]
    cup_in_tcp_pick = np.linalg.inv(pick_g)
    cup_in_tcp_place = np.linalg.inv(place_g)

    # The cycle, as in formulation._plan_from / _move_cup:
    # (kind, start, end, lid angle (or arc), contact, cup held (tcp->cup) or None)
    arc_open = np.linspace(0.0, lid_open, N_ARC)
    # (kind, start, end, lid angle, contact, cup held (tcp->cup) or None, IK branch:
    #  "A"/"B", or (start, end) for a joint move that changes configuration)
    moves = [
        ("joint", ready, a_closed, 0.0, False, None, "A"),
        ("line", a_closed, g_closed, 0.0, True, None, "A"),
        ("arc", arc_open, None, None, True, None, "A"),
        ("line", g_open, a_open, lid_open, True, None, "A"),
        ("joint", a_open, pick_up, lid_open, False, None, "A"),
        ("line", pick_up, pick, lid_open, True, None, "A"),
        ("line", pick, pick_up, lid_open, True, cup_in_tcp_pick, "A"),
        ("joint", pick_up, high, lid_open, False, cup_in_tcp_pick, ("A", "B")),
        ("joint", high, over, lid_open, False, cup_in_tcp_place, "B"),
        ("line", over, pre, lid_open, False, cup_in_tcp_place, "B"),
        ("line", pre, place, lid_open, True, cup_in_tcp_place, "B"),
        ("line", place, pre, lid_open, True, None, "B"),
        ("line", pre, over, lid_open, False, None, "B"),
        ("joint", over, a_open, lid_open, False, None, ("B", "A")),
        ("line", a_open, g_open, lid_open, True, None, "A"),
        ("arc", arc_open[::-1], None, None, True, None, "A"),
        ("line", g_closed, a_closed, 0.0, True, None, "A"),
        ("joint", a_closed, ready, 0.0, False, None, "A"),
    ]
    # Every IK target, solved in one batch per branch.
    targets, groups, spans = [], [], []
    for kind, a, b, lid, contact, cup, branch in moves:
        if kind == "joint":
            poses = np.stack([a, b])
            groups.extend(branch if isinstance(branch, tuple) else (branch, branch))
        elif kind == "line":
            poses, length, angle = line_poses(a, b, N_LINE)
            groups.extend([branch] * len(poses))
        else:
            poses = np.stack([handle_grasp(t, tilt) for t in a])
            groups.extend([branch] * len(poses))
        spans.append((len(targets), len(targets) + len(poses)))
        targets.extend(poses)
    targets, groups = np.stack(targets), np.array(groups)
    q_all, deficit = np.empty((len(targets), 6)), np.empty(len(targets))
    for name, flags in (("A", branch_a), ("B", branch_b)):
        sel = groups == name
        q_all[sel], deficit[sel] = ik(base_inv, targets[sel], *flags)
    out["reach_deficit"] = float(deficit.sum())
    if out["reach_deficit"] > 0 or out["footprint_overlap"] > 0:
        out.update(cycle_time=float("nan"), min_clearance=float("nan"), collision=float("nan"),
                   ik_jumps=float("nan"), manip_shortfall=float("nan"), penalty=float("nan"))
        out["objective"] = W_INFEASIBLE + W_REACH * (out["reach_deficit"] + out["footprint_overlap"])
        out["feasible"] = False
        return out

    # Timing (E7) and the configurations to check for collision.
    t_total = float(sum(GEO["planner"]["gripper_seconds"]))
    jumps, manip_short = 0.0, 0.0
    check_q, check_lid, check_contact, check_cup = [], [], [], []
    prev = np.zeros(6)
    for (kind, a, b, lid, contact, cup, _), (i0, i1) in zip(moves, spans):
        q = unwrap(q_all[i0:i1], prev)
        if kind == "joint":
            t_total += joint_time(q[0], q[1])
            s = np.linspace(0, 1, N_JOINT)[:, None]
            q_s = (1 - s) * q[0] + s * q[1]
            lids = np.full(N_JOINT, lid)
        else:
            dq = np.abs(np.diff(q, axis=0))
            jumps += float(np.maximum(dq.max(1) - JUMP, 0.0).sum())
            q_path = dq.sum(0)
            if kind == "line":
                length = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
                t_total += max(float(trap(length, V_LIN, A_LIN)), float(trap(q_path, V_JOINT, A_JOINT).max()))
                lids = np.full(len(q), lid)
            else:
                tcp_path = np.stack([handle_grasp(t, tilt)[:3, 3] for t in a])
                arc_len = float(np.linalg.norm(np.diff(tcp_path, axis=0), axis=1).sum())
                t_total += max(float(trap(arc_len, V_HANDLE, A_HANDLE)), float(trap(q_path, V_JOINT, A_JOINT).max()))
                lids = a
                w = manipulability(base, q)
                manip_short += float(np.maximum(1.0 - w / W_MIN, 0.0).sum())
            q_s = q
        check_q.append(q_s)
        check_lid.append(lids)
        check_contact.append(np.full(len(q_s), contact))
        if cup is None:
            check_cup.append(np.full((len(q_s), 4, 4), np.nan))
        else:
            tcp_frames = fk(base, q_s)[0]["tool0"] @ _trans(0, 0, TCP)
            check_cup.append(tcp_frames @ cup)
        prev = q[-1]
    d = clearances(
        base,
        np.concatenate(check_q),
        np.concatenate(check_lid),
        np.concatenate(check_contact),
        np.concatenate(check_cup),
        cnc_xy,
    )
    out["cycle_time"] = t_total
    out["min_clearance"] = float(d.min())
    out["collision"] = float(np.maximum(C_MIN - d, 0.0).sum())
    out["ik_jumps"] = jumps
    out["manip_shortfall"] = manip_short
    penalty = W_COLLIDE * out["collision"] + W_JUMP * jumps + W_MANIP * manip_short
    out["penalty"] = P_SAT * (1.0 - math.exp(-penalty / P_SAT))
    out["objective"] = t_total + out["penalty"]
    out["feasible"] = out["collision"] == 0 and jumps == 0 and manip_short == 0
    return out


def objective(x) -> float:
    return float(evaluate(x)["objective"])
