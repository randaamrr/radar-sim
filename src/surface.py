import numpy as np

from src.room import build_room
from src.materials import apply_materials
from src.meshes import occluder_scene, visible

# Outward normals for furniture faces, inward normals for the room shell.
NORMALS = {
    "floor": (0, 0, 1), "ceiling": (0, 0, -1),
    "wall_x0": (1, 0, 0), "wall_x1": (-1, 0, 0),
    "wall_y0": (0, 1, 0), "wall_y1": (0, -1, 0),
    "top": (0, 0, 1), "front": (0, -1, 0), "back": (0, 1, 0),
    "left": (-1, 0, 0), "right": (1, 0, 0),
}


def surface_normals(names):
    out = np.zeros((len(names), 3))
    for s in np.unique(names):
        key = s if s in NORMALS else s.rsplit("_", 1)[1]
        out[names == s] = NORMALS[key]
    return out


def scene_samples(cfg):
    """Surface samples with material strength and normals (room, furniture, people)."""
    room = apply_materials(build_room(cfg))
    room["normal"] = surface_normals(room["surface"])
    from src.people import build_people
    people = build_people(cfg)
    if people is not None:
        room = {k: np.concatenate([room[k], people[k]]) for k in room}
    return room


def radar_view(room, cfg, radar_pos, facing=True, occlusion=True):
    """Copy of the samples with 'strength' as seen from one radar."""
    p = np.asarray(radar_pos, dtype=float)
    vec = p - room["points"]
    d = np.maximum(np.linalg.norm(vec, axis=1), 0.1)
    w = room["strength"].copy()
    if facing:
        cos = np.einsum("ij,ij->i", room["normal"], vec) / d
        w = w * np.clip(cos, 0.0, None)
    if occlusion:
        w = w * visible(occluder_scene(cfg), p, room["points"])
    out = dict(room)
    out["strength"] = w
    return out


def direct_power(room, radar_pos):
    d = np.maximum(np.linalg.norm(room["points"] - np.asarray(radar_pos), axis=1), 0.1)
    return float((room["strength"] / d ** 4).sum())

def removed_by_range(room, cfg, radar_pos, edges=(0, 1, 2, 3, 4, 6)):
    """Percent of facing-only power removed by occlusion, per range window."""
    pos = np.asarray(radar_pos, dtype=float)
    facing = radar_view(room, cfg, pos, occlusion=False)
    both = radar_view(room, cfg, pos)
    d = np.linalg.norm(room["points"] - pos, axis=1)
    dd = np.maximum(d, 0.1)
    pf = facing["strength"] / dd ** 4
    pb = both["strength"] / dd ** 4
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        total = pf[m].sum()
        out.append((lo, hi, 0.0 if total == 0 else 100.0 * (1 - pb[m].sum() / total)))
    return out