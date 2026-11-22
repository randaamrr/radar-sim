import numpy as np
import trimesh

RT = 0.085          # torso radius, fraction of height
SH = 0.115          # shoulder half-width
HP = 0.055          # hip half-width

# (name, joint a, joint b, radius) -- radius as a fraction of body height
CAPSULES = [
    ("torso", "pelvis", "chest", RT),
    ("head", "neck", "head", 0.055),
    ("arm_up_l", "shoulder_l", "elbow_l", 0.028),
    ("arm_up_r", "shoulder_r", "elbow_r", 0.028),
    ("forearm_l", "elbow_l", "wrist_l", 0.022),
    ("forearm_r", "elbow_r", "wrist_r", 0.022),
    ("thigh_l", "hip_l", "knee_l", 0.048),
    ("thigh_r", "hip_r", "knee_r", 0.048),
    ("shin_l", "knee_l", "ankle_l", 0.035),
    ("shin_r", "knee_r", "ankle_r", 0.035),
]
POSES = ("stand", "sit", "lie")
CAP_RINGS, CAP_SEGMENTS = 8, 16   # capsule tessellation (speed vs roundness)


def joints_for(pose, height):
    """Joint positions in the body frame (x forward, y left, z up).

    stand: feet near z=0. sit: lowest point of the pelvis at z=0 (the seat).
    lie: supine on a surface, head toward +x. Segment lengths are rough
    anthropometric fractions of height.
    """
    H = float(height)
    s = lambda x, y, z: np.array([x, y, z], dtype=float) * H
    j = {}
    if pose in ("stand", "sit"):
        z0 = 0.0 if pose == "stand" else RT - 0.53    # sit: pelvis bottom at 0
        j["pelvis"] = s(0, 0, 0.53 + z0)
        j["chest"] = s(0, 0, 0.72 + z0)
        j["neck"] = s(0, 0, 0.82 + z0)
        j["head"] = s(0, 0, 0.945 + z0)
        for side, sg in (("l", 1), ("r", -1)):
            j[f"shoulder_{side}"] = s(0, sg * SH, 0.80 + z0)
            j[f"hip_{side}"] = s(0, sg * HP, 0.53 + z0)
            j[f"elbow_{side}"] = j[f"shoulder_{side}"] + s(0, 0, -0.186)
            if pose == "stand":
                j[f"wrist_{side}"] = j[f"elbow_{side}"] + s(0, 0, -0.20)
                j[f"knee_{side}"] = j[f"hip_{side}"] + s(0, 0, -0.245)
            else:
                j[f"wrist_{side}"] = j[f"elbow_{side}"] + s(0.20, 0, 0)
                j[f"knee_{side}"] = j[f"hip_{side}"] + s(0.245, 0, 0)
            j[f"ankle_{side}"] = j[f"knee_{side}"] + s(0, 0, -0.246)
    elif pose == "lie":
        j["pelvis"] = s(0, 0, 0)
        j["chest"] = s(0.19, 0, 0)
        j["neck"] = s(0.29, 0, 0)
        j["head"] = s(0.415, 0, 0)
        for side, sg in (("l", 1), ("r", -1)):
            j[f"shoulder_{side}"] = s(0.27, sg * SH, 0)
            j[f"elbow_{side}"] = j[f"shoulder_{side}"] + s(-0.186, 0, 0)
            j[f"wrist_{side}"] = j[f"elbow_{side}"] + s(-0.20, 0, 0)
            j[f"hip_{side}"] = s(0, sg * HP, 0)
            j[f"knee_{side}"] = j[f"hip_{side}"] + s(-0.245, 0, 0)
            j[f"ankle_{side}"] = j[f"knee_{side}"] + s(-0.246, 0, 0)
    else:
        raise ValueError(f"unknown pose: {pose}")
    return j


def _capsule(a, b, radius):
    """Capsule mesh whose two sphere centers are the points a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    m = trimesh.creation.capsule(height=max(length, 1e-6), radius=radius,
                                   count=[CAP_RINGS, CAP_SEGMENTS])
    m.apply_translation(-m.bounds.mean(axis=0))          # center at origin
    m.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], (b - a) / length))
    m.apply_translation((a + b) / 2.0)
    return m


def person_meshes(pose, height, x, y, yaw_deg=0.0, support_z=0.0):
    """List of (part name, mesh) for one person placed in the room.

    (x, y): body origin on the floor plane; yaw_deg: heading (0 = facing +x);
    support_z: height of the surface the body rests on (floor 0, bed 0.5...).
    """
    j = joints_for(pose, height)
    parts = [(n, _capsule(j[a], j[b], r * height)) for n, a, b, r in CAPSULES]
    shift = 0.0 if pose == "sit" else -min(m.bounds[0, 2] for _, m in parts)
    rot = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1])
    for _, m in parts:
        m.apply_translation([0, 0, shift])
        m.apply_transform(rot)
        m.apply_translation([x, y, support_z])
    return parts


def merged(parts):
    return trimesh.util.concatenate([m for _, m in parts])


def validate_placement(cfg, parts, tol=0.01):
    """Physical validity only: returns a list of problems (empty = valid).

    Checks the body is inside the room and does not penetrate furniture by
    more than tol. It does NOT check that something supports the body.
    """
    V = np.vstack([m.vertices for _, m in parts])
    size = np.array(cfg["room"]["size_m"], dtype=float)
    problems = []
    if (V < -tol).any() or (V > size + tol).any():
        problems.append("part of the body is outside the room")
    for f in cfg["furniture"]:
        x, y = f["pos_m"]
        w, d, h = f["size_m"]
        lo = np.array([x, y, 0.0]) + tol
        hi = np.array([x + w, y + d, h]) - tol
        if ((V > lo) & (V < hi)).all(axis=1).any():
            problems.append(f"body penetrates {f['name']}")
    return problems


# ---------------------------------------------------------------------------
# Step 22: people inside a scenario
# ---------------------------------------------------------------------------
HEIGHT_RANGE_M = (1.0, 2.1)    # assumption: adults and tall children only


def cfg_people(cfg):
    """[(person name, [(part, mesh), ...])] for every person in the scenario."""
    out = []
    for p in cfg.get("people", []):
        x, y = p["pos_m"]
        parts = person_meshes(p["pose"], p["height_m"], x, y,
                              p.get("yaw_deg", 0.0), p.get("support_z", 0.0))
        out.append((p["name"], parts))
    return out


def check_support(cfg, x, y, support_z, tol=0.02):
    """True if support_z is the floor, or the top of a furniture item whose
    footprint contains the body origin (x, y)."""
    if abs(support_z) <= tol:
        return True
    for f in cfg["furniture"]:
        fx, fy = f["pos_m"]
        w, d, h = f["size_m"]
        if (abs(h - support_z) <= tol and fx - tol <= x <= fx + w + tol
                and fy - tol <= y <= fy + d + tol):
            return True
    return False


def bodies_overlap(parts_a, parts_b):
    """True if a vertex of one body lies inside a part of the other."""
    for src, dst in ((parts_a, parts_b), (parts_b, parts_a)):
        V = np.vstack([m.vertices for _, m in src])
        for _, m in dst:
            if m.contains(V).any():
                return True
    return False


def sample_person(name, parts, step, rng):
    """Surface samples of one body (dict with the same keys as the room).

    Each capsule is sampled uniformly by area (about one sample per
    step x step), with the face normal and an equal area share per sample.
    Samples inside another capsule of the same body are dropped, so the
    remaining samples approximate the outer skin. Dropping them also
    removes their area, which is what we want.
    """
    from src.materials import body_reflectance
    chunks = []
    for i, (part, mesh) in enumerate(parts):
        n = max(4, int(round(mesh.area / step ** 2)))
        pts, face = trimesh.sample.sample_surface(mesh, n, seed=rng)
        keep = np.ones(n, dtype=bool)
        for j, (_, other) in enumerate(parts):
            if j != i:
                keep &= ~other.contains(pts)
        chunks.append((part, pts[keep], mesh.face_normals[face][keep],
                       mesh.area / n))
    pts = np.vstack([c[1] for c in chunks])
    out = {
        "points": pts,
        "normal": np.vstack([c[2] for c in chunks]),
        "area": np.concatenate([np.full(len(c[1]), c[3]) for c in chunks]),
        "surface": np.array([f"{name}_{c[0]}" for c in chunks
                             for _ in range(len(c[1]))]),
    }
    out["material"] = np.full(len(pts), "skin")
    out["reflectance"] = np.full(len(pts), body_reflectance())
    out["strength"] = out["area"] * out["reflectance"]
    return out


def build_people(cfg):
    """Samples of every person in the scenario, or None if there are none."""
    people = cfg_people(cfg)
    if not people:
        return None
    step = cfg["room"]["grid_step_m"]
    chunks = []
    for k, (name, parts) in enumerate(people):
        rng = np.random.default_rng([int(cfg["seed"]), k])
        chunks.append(sample_person(name, parts, step, rng))
    return {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
