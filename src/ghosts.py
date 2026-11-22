import numpy as np

from src.materials import reflectance
from src.meshes import occluder_scene, blocked_segments
from src.multipath import _planes, mirror


def segments_clear(scene, starts, ends, eps=1e-3):
    """True where the straight segment start->end is not blocked."""
    return ~blocked_segments(scene, starts, ends, eps)


def first_order_ghosts_view(room, cfg, radar_pos, facing=True, occlusion=True):
    """One-bounce ghosts (image method) with incidence weighting and occlusion.

    room must come from scene_samples (needs 'normal'). With facing=False and
    occlusion=False this reproduces multipath.first_order_ghosts exactly.
    """
    radar = np.asarray(radar_pos, dtype=float)
    pts, nrm = room["points"], room["normal"]
    scene = occluder_scene(cfg) if occlusion else None

    dist = np.linalg.norm(pts - radar, axis=1)
    dr = np.maximum(dist, 0.1)
    c1 = np.einsum("ij,ij->i", nrm, radar - pts) / np.maximum(dist, 1e-9)
    direct_ok = segments_clear(scene, np.tile(radar, (len(pts), 1)), pts)

    ranges, powers = [], []
    for axis, coord, mat in _planes(cfg).values():
        if abs(radar[axis] - coord) < 1e-6:
            continue                      # radar is mounted on this plane
        img = mirror(radar, axis, coord)
        keep = direct_ok & (np.abs(pts[:, axis] - coord) > 1e-6)
        if facing:
            keep &= c1 > 0
        idx = np.nonzero(keep)[0]
        p = pts[idx]
        t = (coord - img[axis]) / (p[:, axis] - img[axis])
        bounce = img + t[:, None] * (p - img)

        w = np.ones(len(idx))
        if facing:
            to_b = bounce - p
            c2 = (np.einsum("ij,ij->i", nrm[idx], to_b)
                  / np.maximum(np.linalg.norm(to_b, axis=1), 1e-9))
            w = np.sqrt(np.clip(c1[idx], 0, None) * np.clip(c2, 0, None))
            ok = w > 0
            idx, p, bounce, w = idx[ok], p[ok], bounce[ok], w[ok]
        if occlusion and len(idx):
            clear = (segments_clear(scene, np.tile(radar, (len(idx), 1)), bounce)
                     & segments_clear(scene, bounce, p))
            idx, p, w = idx[clear], p[clear], w[clear]

        d = dr[idx]
        d_img = np.maximum(np.linalg.norm(p - img, axis=1), 0.1)
        ranges.append((d + d_img) / 2.0)
        powers.append(2.0 * room["strength"][idx] * reflectance(mat) * w
                      / (d ** 2 * d_img ** 2))
    return np.concatenate(ranges), np.concatenate(powers)