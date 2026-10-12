# SPDX-License-Identifier: MIT
"""Read one PandaSet sequence as distributed (camera/, lidar/, annotations/):
JPEG frames, camera poses and intrinsics, LiDAR sweeps and 3D cuboids.

PandaSet conventions used here:
- LiDAR points are already in the world frame (meters, Z up).
- A pose is {"position": {x, y, z}, "heading": {w, x, y, z}}: the sensor-to-world
  transform. Cameras use the OpenCV convention (+Z forward, +X right, +Y down),
  the same one the projection in splat/render.py uses.
- A cuboid has a center, dimensions (x = width, y = length, z = height) and a
  yaw about world Z; the car's length runs along its local y axis.
"""
import json
import math
import os

import numpy as np

CAMERAS = ("front_camera", "front_left_camera", "front_right_camera")

# Objects that are masked out of training even when they don't move: people
# (privacy, and they are never perfectly still).
ALWAYS_MASK = ("Pedestrian", "Pedestrian with Object")
VEHICLES = ("Car", "Pickup Truck", "Medium-sized Truck", "Semi-truck", "Bus", "Emergency Vehicle",
            "Other Vehicle - Uncommon", "Other Vehicle - Construction Vehicle", "Other Vehicle - Pedicab",
            "Towed Object", "Motorcycle")


def pose_to_matrix(pose: dict) -> np.ndarray:
    """PandaSet pose -> 4x4 sensor-to-world matrix."""
    w, x, y, z = (pose["heading"][k] for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [pose["position"][k] for k in ("x", "y", "z")]
    return m


def load_camera(seq_dir: str, camera: str) -> dict:
    d = os.path.join(seq_dir, "camera", camera)
    with open(os.path.join(d, "intrinsics.json")) as f:
        k = json.load(f)
    with open(os.path.join(d, "poses.json")) as f:
        poses = json.load(f)
    with open(os.path.join(d, "timestamps.json")) as f:
        stamps = json.load(f)
    K = np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])
    return {"K": K, "c2w": np.stack([pose_to_matrix(p) for p in poses]), "timestamps": np.array(stamps),
            "image_path": lambda f: os.path.join(d, f"{f:02d}.jpg")}


def load_lidar_poses(seq_dir: str) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(seq_dir, "lidar", "poses.json")) as f:
        poses = json.load(f)
    with open(os.path.join(seq_dir, "lidar", "timestamps.json")) as f:
        stamps = json.load(f)
    return np.stack([pose_to_matrix(p) for p in poses]), np.array(stamps)


def load_lidar(seq_dir: str, frame: int) -> dict:
    """One sweep of both LiDARs: d = 0 is the 360-degree Pandar64 on the roof,
    d = 1 the forward-facing PandarGT."""
    import pandas as pd                        # only needed to read the raw PandaSet pickles
    df = pd.read_pickle(os.path.join(seq_dir, "lidar", f"{frame:02d}.pkl"))
    return {"xyz": df[["x", "y", "z"]].to_numpy(np.float64), "intensity": df["i"].to_numpy(np.float32),
            "sensor": df["d"].to_numpy(np.int8)}


def load_cuboids(seq_dir: str, frame: int) -> list[dict]:
    import pandas as pd
    df = pd.read_pickle(os.path.join(seq_dir, "annotations", "cuboids", f"{frame:02d}.pkl"))
    center = df[["position.x", "position.y", "position.z"]].to_numpy(float)
    size = df[["dimensions.x", "dimensions.y", "dimensions.z"]].to_numpy(float)
    return [{"uuid": u, "label": lab, "stationary": bool(st), "center": c.tolist(), "size": s.tolist(),
             "yaw": float(yaw)}
            for u, lab, st, c, s, yaw in zip(df["uuid"], df["label"], df["stationary"], center, size, df["yaw"])]


# --- boxes -----------------------------------------------------------------

def box_corners(box: dict, grow: float | tuple = 0.0) -> np.ndarray:
    """The 8 world-space corners (8,3); `grow` meters added on every side
    (one number, or per axis: width, length, height)."""
    g = np.broadcast_to(np.asarray(grow, float), (3,))
    hx, hy, hz = (np.asarray(box["size"]) / 2 + g)
    local = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ rot.T + np.asarray(box["center"])


def points_in_box(xyz: np.ndarray, box: dict, grow: float = 0.0) -> np.ndarray:
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    d = xyz - np.asarray(box["center"])
    lx = c * d[:, 0] + s * d[:, 1]
    ly = -s * d[:, 0] + c * d[:, 1]
    half = np.asarray(box["size"]) / 2 + grow
    return (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1]) & (np.abs(d[:, 2]) <= half[2])


BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def project_polygon(corners: np.ndarray, w2c: np.ndarray, K: np.ndarray, near: float = 0.1,
                    edges=BOX_EDGES) -> np.ndarray | None:
    """Image-space convex hull of a 3D box (or any corner set with edges),
    clipped at the near plane so boxes beside the camera still project
    correctly. Returns (k,2) pixel polygon or None."""
    pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
    pts = [p for p in pc if p[2] > near]
    for i, j in edges:                          # where an edge crosses the near plane
        zi, zj = pc[i, 2], pc[j, 2]
        if (zi - near) * (zj - near) < 0:
            t = (near - zi) / (zj - zi)
            pts.append(pc[i] + t * (pc[j] - pc[i]))
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    uv = pts[:, :2] / pts[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return convex_hull(uv)


def convex_hull(p: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, (k,2) counter-clockwise."""
    p = np.unique(p, axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(points):
        h = []
        for q in points:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]
    return np.array(half(p) + half(p[::-1]))


def track_boxes(per_frame: list[list[dict]], lidar_times: np.ndarray, move_thresh: float = 0.5) -> dict:
    """Group boxes by object id over the sequence. Returns uuid -> {label,
    frames, centers (F,3), sizes, yaws, dynamic}. An object is dynamic if its
    center wanders more than `move_thresh` meters from its median position,
    or if it is a person."""
    tracks = {}
    for f, boxes in enumerate(per_frame):
        for b in boxes:
            t = tracks.setdefault(b["uuid"], {"label": b["label"], "frames": [], "centers": [], "sizes": [],
                                              "yaws": []})
            if t["frames"] and t["frames"][-1] == f:     # duplicate box from the second LiDAR
                continue
            t["frames"].append(f)
            t["centers"].append(b["center"])
            t["sizes"].append(b["size"])
            t["yaws"].append(b["yaw"])
    for t in tracks.values():
        c = np.asarray(t["centers"])
        wander = np.linalg.norm(c - np.median(c, axis=0), axis=1).max()
        t["dynamic"] = bool(wander > move_thresh or t["label"] in ALWAYS_MASK)
        t["times"] = lidar_times[t["frames"]].tolist()
    return tracks


def box_at_time(track: dict, t: float) -> dict | None:
    """The object's box linearly interpolated to time t (cameras fire 40-60 ms
    before the LiDAR timestamp, which a 10 m/s car covers in about half a meter).
    None if t is outside the object's annotated span (beyond half a frame)."""
    times = np.asarray(track["times"])
    if t < times[0] - 0.05 or t > times[-1] + 0.05:
        return None
    c = np.asarray(track["centers"])
    center = [float(np.interp(t, times, c[:, k])) for k in range(3)]
    yaws = np.unwrap(np.asarray(track["yaws"]))
    return {"label": track["label"], "center": center, "size": list(np.median(track["sizes"], axis=0)),
            "yaw": float(np.interp(t, times, yaws))}
