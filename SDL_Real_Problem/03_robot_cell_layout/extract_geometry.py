"""Export the SDL5 formulation cell's real geometry from the sdl_dt digital twin.

Reads the sdl_dt package (UR3e and ARE-312 URDFs, packed CAD boxes, planner
constants in src/are312/formulation.py) and writes data/cell_geometry.json,
which is all model.py needs, so the task runs without sdl_dt installed.

Usage (needs a checkout of sdl_dt):
    python extract_geometry.py --sdl-dt C:/Users/mrast/Github_Mahdi/sdl_dt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fit_spheres(points: np.ndarray, k: int = 6, seed: int = 0) -> dict:
    """Bounding spheres of a link's mesh vertices (link frame): k-means clusters
    of the vertices, each enclosed by the sphere about its centroid through its
    farthest vertex, so every vertex lies inside some sphere."""
    rng = np.random.default_rng(seed)
    k = min(k, len(points))
    centres = points[rng.choice(len(points), k, replace=False)]
    for _ in range(50):
        label = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[label == j].mean(0) if (label == j).any() else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    label = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    keep = [j for j in range(k) if (label == j).any()]
    radii = [float(np.linalg.norm(points[label == j] - centres[j], axis=1).max()) for j in keep]
    return {"centres": centres[keep].tolist(), "radii": radii}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdl-dt", default="C:/Users/mrast/Github_Mahdi/sdl_dt")
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.sdl_dt) / "src"))
    from are312 import formulation as F
    from are312.arm import UR3E_JOINT_LIMITS
    from are312.description.kinematics import link_transforms

    src = "sdl_dt/src/are312/formulation.py"

    # UR3e chain as URDF joint origins (each revolute joint rotates about its local z).
    chain = []
    for joint, origin in zip(F._UR3E_CHAIN, F._UR3E_ORIGINS):
        chain.append({"name": joint.name, "child": joint.child, "kind": joint.kind, "origin": origin.tolist()})
    spheres = {link: fit_spheres(points) for link, points in F._UR3E_VERTICES}

    # Mixer: lid hinge (door_joint), lid handle, bowl seat. Check the closed form
    # cover(a) = Trans(0, 0, 0.328) Rot(-y, a) against sdl_dt's link_transforms.
    door = next(j for j in F.ARE312_URDF.joints if j.name == "door_joint")
    grasp_fixed = next(j for j in F.ARE312_URDF.joints if j.child == "thinky_grasp_cover")
    for a in (0.0, 0.6, F.LID_OPEN_ANGLE):
        t = link_transforms(F.ARE312_URDF, {"door_joint": a, "revolver_joint": F.MIXER_BOWL_ANGLE})
        c, s = math.cos(a), math.sin(a)
        ours = np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0.328], [0, 0, 0, 1.0]])
        assert np.allclose(np.array(t["thinky_cover"]), ours, atol=1e-9), "lid hinge model mismatch"

    cnc_boxes = F._cnc_boxes()  # CNC frame, axes parked
    scale_boxes = F._scale_boxes_in_cnc()
    ws = np.load(Path(args.sdl_dt) / "src/are312/mesh/workstation.npz")
    table = ws["boxes"][ws["box_links"] == "table"]
    wsp = F.FORMULATION_WORKSTATION_POSE  # yaw +90 deg: cell x = x0 - y_ws, cell y = y0 + x_ws
    deck = {
        "x": [wsp.xyz[0] - float(table[:, 1, 1].max()), wsp.xyz[0] - float(table[:, 0, 1].min())],
        "y": [wsp.xyz[1] + float(table[:, 0, 0].min()), wsp.xyz[1] + float(table[:, 1, 0].max())],
    }
    cup_on_scale = np.linalg.inv(F._matrix(F.FORMULATION_CNC_POSE)) @ F._cup_on_scale(F.FORMULATION_CNC_POSE)

    geo = {
        "_provenance": "Exported from the sdl_dt digital twin (SDL5 formulation cell) by extract_geometry.py. "
        "Frames: cell frame = ARE-312 mixer base frame (z up, deck at z = 0). Units: metres, radians.",
        "ur3e": {
            "source": "sdl_dt/src/are312/description/ur3e.urdf via formulation._UR3E_CHAIN; joint limits from arm.UR3E_JOINT_LIMITS",
            "chain": chain,
            "joint_limits": [list(l) for l in UR3E_JOINT_LIMITS],
            "dh": {
                "a": [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0],
                "d": [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921],
                "alpha": [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0],
                "note": "standard UR3e DH; verified: T_urdf_tool0(q) = Rz(pi) @ T_dh(q) (see build_data.py validation)",
            },
            "tcp_offset": F.TCP_OFFSET,
            "link_spheres": spheres,
            "spheres_note": "bounding spheres of each link's visual-mesh vertices (formulation._UR3E_VERTICES), k-means (k = 6) clusters, each enclosed by one sphere, link frame",
            "plate_height": F.PLATE_HEIGHT,
            "base_footprint_radius": 0.064,
            "base_footprint_note": "UR3e base flange is 128 mm in diameter (UR3e datasheet)",
        },
        "gripper": {
            "source": src + " (GRIPPER_BODY_LENGTH, GRIPPER_FINGER_LENGTH, _gripper_points)",
            "body_length": F.GRIPPER_BODY_LENGTH,
            "finger_length": F.GRIPPER_FINGER_LENGTH,
            "body_radius": math.hypot(0.035, 0.03),
            "finger_radius": F.GRIPPER_STROKE / 2 + 0.012,
        },
        "mixer": {
            "source": "sdl_dt/src/are312/description/are312.urdf; " + src + " (_BODY_BOX, _LID_BOX, _MIXER_CAVITY)",
            "body_box": [list(F._BODY_BOX[1]), list(F._BODY_BOX[2])],
            "lid_box": [list(F._LID_BOX[1]), list(F._LID_BOX[2])],
            "lid_hinge_z": float(door.pose.xyz[2]),
            "lid_hinge_axis": list(door.axis),
            "lid_closed": F.LID_CLOSED_ANGLE,
            "lid_open_max": F.LID_OPEN_ANGLE,
            "handle_in_lid": list(grasp_fixed.pose.xyz),
            "cavity": list(F._MIXER_CAVITY),
            "cup_seat": F._cup_seat().tolist(),
            "cup_seat_note": "thinky_cup frame with revolver_joint at MIXER_BOWL_ANGLE = -115 deg",
        },
        "cup": {
            "source": src + " (CUP_*, _cup_grasp, _CUP_LIFT, _CUP_STANDOFF)",
            "radius": F.CUP_RADIUS,
            "wall": F.CUP_WALL,
            "height": F.CUP_HEIGHT,
            "grasp_depth": F._CUP_GRASP_DEPTH,
            "grasp_tilt": F.CUP_GRASP_TILT,
            "place_azimuth": F.CUP_PLACE_AZIMUTH,
            "release_height": F.CUP_RELEASE_HEIGHT,
            "lift": F._CUP_LIFT,
            "standoff": F._CUP_STANDOFF,
        },
        "cnc": {
            "source": "sdl_dt/src/are312/mesh/cnc.npz and scale.npz via formulation._cnc_boxes(), _scale_boxes_in_cnc(), _cup_on_scale()",
            "boxes": np.concatenate([cnc_boxes, scale_boxes]).tolist(),
            "n_frame_boxes": int(len(cnc_boxes)),
            "footprint": [cnc_boxes[:, 0, :2].min(axis=0).tolist(), cnc_boxes[:, 1, :2].max(axis=0).tolist()],
            "cup_on_scale": cup_on_scale.tolist(),
            "cup_on_scale_note": "cup frame on the balance next to the arm, in the CNC frame",
        },
        "deck": {
            "source": "sdl_dt/src/are312/mesh/workstation.npz table boxes at FORMULATION_WORKSTATION_POSE",
            **deck,
        },
        "default_layout": {
            "source": src + " (FORMULATION_BASE_POSE, FORMULATION_CNC_POSE, FORMULATION_GRASP_TILT, LID_OPEN_ANGLE)",
            "base_xy": list(F.FORMULATION_BASE_POSE.xyz[:2]),
            "base_yaw": F.FORMULATION_BASE_POSE.rpy[2],
            "cnc_xy": list(F.FORMULATION_CNC_POSE.xyz[:2]),
            "lid_open": F.LID_OPEN_ANGLE,
            "grasp_tilt": list(F.FORMULATION_GRASP_TILT),
        },
        "planner": {
            "source": src + " (_APPROACHES, _plan_from ready pose, _CUP_OVER, set_gripper/release durations)",
            "standoff": 0.05,
            "ready_back": 0.30,
            "ready_lift": 0.05,
            "over_rise": 0.08,
            "high_rise": 0.04,
            "gripper_seconds": [0.7, 0.6, 0.7, 0.8, 0.7, 0.6],
            "ik_continuity_rad": 0.25,
            "min_clearance": 0.005,
        },
    }
    out = HERE / "data" / "cell_geometry.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(geo, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
