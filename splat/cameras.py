# SPDX-License-Identifier: MIT
"""Virtual cameras for novel views: look-at, offsets from a recorded pose,
smooth interpolation along the drive. All poses are camera-to-world 4x4 in the
OpenCV convention (columns: right, down, forward, position), world Z up."""
import math

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def look_at(eye, target, up=UP) -> np.ndarray:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w


def offset_pose(c2w: np.ndarray, right: float = 0.0, up: float = 0.0, forward: float = 0.0,
                pitch_down_deg: float = 0.0) -> np.ndarray:
    """Move a camera in road-aligned directions (right and forward level with
    the ground, up = world Z), then tilt it down about its own right axis."""
    out = c2w.copy()
    fwd = c2w[:3, 2].copy()
    fwd_level = np.array([fwd[0], fwd[1], 0.0])
    fwd_level /= np.linalg.norm(fwd_level)
    right_level = np.cross(fwd_level, UP)
    out[:3, 3] = c2w[:3, 3] + right * right_level + up * UP + forward * fwd_level
    if pitch_down_deg:
        c, s = math.cos(math.radians(pitch_down_deg)), math.sin(math.radians(pitch_down_deg))
        tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
        out[:3, :3] = c2w[:3, :3] @ tilt
    return out


def car_right(front_c2w: np.ndarray) -> np.ndarray:
    """The car's rightward axis, level with the ground, from the front camera's
    pose (it looks along the direction of travel)."""
    fwd = front_c2w[:3, 2].copy()
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    return np.cross(fwd, UP)


def shift_rig(c2w: np.ndarray, front_c2w: np.ndarray, right: float = 0.0) -> np.ndarray:
    """Move any camera of the rig sideways with the car: along the car's right
    axis (taken from the front camera at the same moment), orientation unchanged.
    A lane change moves every camera the same way, whatever direction it faces."""
    out = c2w.copy()
    out[:3, 3] = c2w[:3, 3] + right * car_right(front_c2w)
    return out


def _rot_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def _quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def interpolate(c2ws, n_out: int) -> list:
    """n_out poses evenly spaced (in index) along a recorded pose sequence:
    positions linearly, rotations by slerp."""
    c2ws = [np.asarray(c, float) for c in c2ws]
    out = []
    for s in np.linspace(0, len(c2ws) - 1, n_out):
        i = min(int(s), len(c2ws) - 2)
        t = s - i
        a, b = c2ws[i], c2ws[i + 1]
        m = np.eye(4)
        m[:3, :3] = _quat_to_rot(_slerp(_rot_to_quat(a[:3, :3]), _rot_to_quat(b[:3, :3]), t))
        m[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
        out.append(m)
    return out


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)
