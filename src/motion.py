"""Step 23: kinematic motion of the capsule body (forward kinematics).

A pose is a handful of angles (trunk lean, hip, knee, shoulder, elbow).
Segment lengths never change, so a motion is a sequence of poses and every
frame is a valid rigid-limb body. Sequences are generated in a local frame
(start pelvis at x=y=0, heading +x, floor z=0) and placed with place().

Known limits (all stated in the decision note):
  * no foot-contact / IK: a stance foot can slide on the floor (foot skate);
  * walking, sitting and falling are stylized curves, not mocap;
  * the body is a stiff plank while falling (legs rotate with the trunk).
"""
import numpy as np
import trimesh

from src.people import CAPSULES, SH, HP, RT, _capsule

JOINTS = ["pelvis", "chest", "neck", "head",
          "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
          "wrist_l", "wrist_r", "hip_l", "hip_r",
          "knee_l", "knee_r", "ankle_l", "ankle_r"]
IDX = {n: i for i, n in enumerate(JOINTS)}

# Segment lengths as fractions of height (same numbers as people.joints_for).
CHEST, SHOULDER, NECK, HEAD = 0.19, 0.27, 0.29, 0.415
UPPER_ARM, FOREARM, THIGH, SHIN = 0.186, 0.20, 0.245, 0.246

# Assumptions (no measured source found; see the decision note).
BREATH_P2P_MM = 5.0     # peak-to-peak chest wall motion; a simulation paper uses 4-6 mm
BREATH_HZ = 0.25        # 15 breaths/min, inside the 12-20 breaths/min resting range
STEP_RATIO = 0.41       # step length / height (rule of thumb, not verified)


def pose(lean=0.0, az=0.0, hip=(0.0, 0.0), knee=(0.0, 0.0),
         sh=(0.0, 0.0), el=(0.0, 0.0)):
    """Angles in degrees. lean: trunk tilt; az: lean direction measured from
    the body's forward axis (0 fwd, 90 left, 180 back, 270 right). hip, knee,
    sh (shoulder), el (elbow) are (left, right) flexion, positive = forward
    (knee/elbow: the lower segment folds backwards)."""
    return {"lean": lean, "az": az, "hip": hip, "knee": knee, "sh": sh, "el": el}


STAND = pose()
SIT = pose(hip=(90, 90), knee=(90, 90), el=(90, 90))
LIE = pose(lean=90, az=180)       # supine, head toward -x


def trunk_rotation(lean_deg, az_deg):
    """3x3 rotation of the upright body frame (columns: forward, left, up)."""
    a = np.radians(az_deg)
    axis = [-np.sin(a), np.cos(a), 0.0]
    return trimesh.transformations.rotation_matrix(np.radians(lean_deg), axis)[:3, :3]


def _dir(up, fwd, ang_deg):
    a = np.radians(ang_deg)
    return -up * np.cos(a) + fwd * np.sin(a)


def skeleton(H, p):
    """Joint positions (16, 3) in the body frame, pelvis at the origin."""
    R = trunk_rotation(p["lean"], p["az"])
    fwd, left, up = R[:, 0], R[:, 1], R[:, 2]
    J = np.zeros((len(JOINTS), 3))
    J[IDX["chest"]] = up * CHEST * H
    J[IDX["neck"]] = up * NECK * H
    J[IDX["head"]] = up * HEAD * H
    for i, (side, sg) in enumerate((("l", 1.0), ("r", -1.0))):
        shoulder = up * SHOULDER * H + sg * left * SH * H
        elbow = shoulder + UPPER_ARM * H * _dir(up, fwd, p["sh"][i])
        wrist = elbow + FOREARM * H * _dir(up, fwd, p["sh"][i] + p["el"][i])
        hip = sg * left * HP * H
        knee = hip + THIGH * H * _dir(up, fwd, p["hip"][i])
        ankle = knee + SHIN * H * _dir(up, fwd, p["hip"][i] - p["knee"][i])
        for name, v in (("shoulder", shoulder), ("elbow", elbow), ("wrist", wrist),
                        ("hip", hip), ("knee", knee), ("ankle", ankle)):
            J[IDX[f"{name}_{side}"]] = v
    return J


def lowest(J, H):
    """Lowest point of the body (capsule extents), relative to the pelvis."""
    return min(min(J[IDX[a]][2], J[IDX[b]][2]) - r * H for _, a, b, r in CAPSULES)


class Sequence:
    """Joint trajectory of one person. joints: (F, 16, 3) in world metres."""

    def __init__(self, kind, fps, height, joints, breath=None, params=None):
        self.kind, self.fps, self.height = kind, float(fps), float(height)
        self.joints = np.asarray(joints, dtype=float)
        self.breath = np.zeros(len(self.joints)) if breath is None else np.asarray(breath, dtype=float)
        self.params = dict(params or {})

    @property
    def n_frames(self):
        return len(self.joints)

    @property
    def duration(self):
        return (self.n_frames - 1) / self.fps

    def parts(self, k):
        """[(part name, capsule mesh)] at frame k (same topology every frame)."""
        J, H = self.joints[k], self.height
        out = []
        for name, a, b, r in CAPSULES:
            extra = self.breath[k] if name == "torso" else 0.0
            out.append((name, _capsule(J[IDX[a]], J[IDX[b]], r * H + extra)))
        return out

    def copy(self, **kw):
        d = dict(kind=self.kind, fps=self.fps, height=self.height,
                 joints=self.joints.copy(), breath=self.breath.copy(),
                 params=dict(self.params))
        d.update(kw)
        return Sequence(**d)


def _times(duration, fps):
    return np.arange(int(np.floor(duration * fps + 1e-9)) + 1) / fps


def _assemble(H, poses, xy=None, floor=0.0, z=None, pivot=False, clip_floor=False):
    """Poses -> world joints. Heading +x, start pelvis at xy[0].

    z=None: the lowest point of the body rests on `floor` every frame.
    clip_floor=True (with z): z is raised when needed so no part goes below floor.
    pivot=True: the mean ankle position stays where it was in frame 0 (a
    body that rotates about its feet, used for falls)."""
    F = len(poses)
    out = np.zeros((F, len(JOINTS), 3))
    xy = np.zeros((F, 2)) if xy is None else np.asarray(xy, dtype=float)
    anchor = None
    for k, p in enumerate(poses):
        J = skeleton(H, p)
        zg = floor - lowest(J, H)
        zk = zg if z is None else (max(z[k], zg) if clip_floor else z[k])
        ank = J[[IDX["ankle_l"], IDX["ankle_r"]], :2].mean(axis=0)
        if pivot:
            if anchor is None:
                anchor = ank + xy[0]
            base = anchor - ank
        else:
            base = xy[k]
        out[k, :, :2] = J[:, :2] + base
        out[k, :, 2] = J[:, 2] + zk
    return out


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return 3 * u ** 2 - 2 * u ** 3


# --------------------------------------------------------------------------
# Motions
# --------------------------------------------------------------------------
def still(pose_name, H, fps, duration):
    p = {"stand": STAND, "sit": SIT, "lie": LIE}[pose_name]
    t = _times(duration, fps)
    if pose_name == "sit":
        J = skeleton(H, p)
        z = np.full(len(t), RT * H)
        joints = _assemble(H, [p] * len(t), z=z)
    else:
        joints = _assemble(H, [p] * len(t))
    return Sequence(f"{pose_name}_still", fps, H, joints, params={"duration": duration})


def walk(H, fps, duration, speed=1.31, step_ratio=STEP_RATIO, knee_flex=55.0,
         arm_swing=15.0, lean=3.0):
    """Stylized gait on a straight line at constant speed (default speed:
    pooled usual-pace mean of healthy adults, 1.31 m/s)."""
    t = _times(duration, fps)
    step = step_ratio * H
    f_cycle = speed / (2.0 * step)                       # strides per second
    amp = np.degrees(np.arcsin(min(step / (2.0 * (THIGH + SHIN) * H), 0.99)))
    knee = lambda ph: 5.0 + knee_flex * max(0.0, np.cos(ph + 0.35)) ** 2
    poses = []
    for ph in 2 * np.pi * f_cycle * t:
        s = np.sin(ph)
        poses.append(pose(lean=lean, hip=(lean + amp * s, lean - amp * s),
                          knee=(knee(ph), knee(ph + np.pi)),
                          sh=(lean - arm_swing * s, lean + arm_swing * s), el=(20.0, 20.0)))
    xy = np.c_[speed * t, np.zeros_like(t)]
    return Sequence("walk", fps, H, _assemble(H, poses, xy),
                    params={"speed": speed, "duration": duration, "f_cycle": f_cycle,
                            "hip_amp_deg": amp})


def sit_down(H, fps, duration=2.4, seat_z=0.5, back_m=0.25, lean_peak=25.0):
    """Stand -> sit on a seat whose top is at seat_z. The pelvis moves back
    by back_m while it lowers onto the seat."""
    t = _times(duration, fps)
    u = t / duration
    s = _smooth(u)
    # Hip and shoulder angles are measured from the trunk, so the trunk lean is
    # added to keep thighs and arms hanging vertically while the trunk tilts.
    poses = []
    for uu, ss in zip(u, s):
        lean = lean_peak * np.sin(np.pi * np.clip(uu, 0, 1))
        poses.append(pose(lean=lean, hip=(90 * ss + lean,) * 2, knee=(90 * ss,) * 2,
                          sh=(lean, lean), el=(90 * ss, 90 * ss)))
    z0 = -lowest(skeleton(H, STAND), H)
    z1 = seat_z + RT * H
    z = (1 - s) * z0 + s * z1
    xy = np.c_[-back_m * s, np.zeros_like(s)]
    return Sequence("sit", fps, H, _assemble(H, poses, xy, z=z, clip_floor=True),
                    params={"duration": duration, "seat_z": seat_z, "back_m": back_m,
                            "lean_peak": lean_peak})


def stand_up(H, fps, duration=2.4, seat_z=0.5, back_m=0.25, lean_peak=25.0):
    """Reverse of sit_down; the start (seated) pelvis is at x=y=0."""
    s = sit_down(H, fps, duration, seat_z, back_m, lean_peak)
    J = s.joints[::-1].copy()
    J[:, :, :2] -= J[0, IDX["pelvis"], :2]
    return Sequence("stand", fps, H, J, params=s.params)


def fall(H, fps, direction_deg=0.0, duration=0.8, exponent=2.0, knee_buckle=30.0,
         arm_reach=40.0, pre_s=1.0, rest_s=2.0):
    """Stand still pre_s, fall for `duration` seconds, then lie still rest_s.

    The body pivots about its feet. direction_deg: 0 forward (prone), 180
    backward (supine), +-90 sideways. exponent > 1: the lean accelerates."""
    total = pre_s + duration + rest_s
    t = _times(total, fps)
    u = np.clip((t - pre_s) / duration, 0.0, 1.0)
    poses = []
    for uu in u:
        b = np.sin(np.pi * uu)
        poses.append(pose(lean=90.0 * uu ** exponent, az=direction_deg,
                          hip=(0.5 * knee_buckle * b,) * 2, knee=(knee_buckle * b,) * 2,
                          sh=(arm_reach * b,) * 2, el=(20.0, 20.0)))
    return Sequence("fall", fps, H, _assemble(H, poses, pivot=True),
                    params={"direction_deg": direction_deg, "duration": duration,
                            "exponent": exponent, "knee_buckle": knee_buckle,
                            "arm_reach": arm_reach, "pre_s": pre_s, "rest_s": rest_s})


def with_breathing(seq, hz=BREATH_HZ, p2p_mm=BREATH_P2P_MM, phase=0.0):
    """Chest wall motion as a torso radius that oscillates (peak-to-peak p2p_mm)."""
    t = np.arange(seq.n_frames) / seq.fps
    breath = 0.5 * p2p_mm * 1e-3 * np.sin(2 * np.pi * hz * t + phase)
    out = seq.copy(breath=breath)
    out.params.update({"breath_hz": hz, "breath_p2p_mm": p2p_mm})
    return out


def place(seq, x, y, yaw_deg=0.0, support_z=0.0):
    """Rigid placement of a local-frame sequence into the room."""
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    J = seq.joints.copy()
    J[:, :, :2] = J[:, :, :2] @ R.T + np.array([x, y])
    J[:, :, 2] += support_z
    return seq.copy(joints=J)


def validate_sequence(cfg, seq, tol=0.03):
    """Problems per frame (inside the room, no penetration of furniture).
    tol is looser than for a static pose because limbs move between frames."""
    from src.people import validate_placement
    problems = []
    for k in range(seq.n_frames):
        for msg in validate_placement(cfg, seq.parts(k), tol=tol):
            problems.append(f"frame {k} (t={k / seq.fps:.2f} s): {msg}")
    return problems


def vertex_tracks(seq):
    """Vertices of every part over time: (F, P, V, 3). Same topology each frame,
    so differences between frames are true surface velocities."""
    return np.stack([np.stack([m.vertices for _, m in seq.parts(k)])
                     for k in range(seq.n_frames)])
