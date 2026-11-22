import numpy as np

from src.materials import reflectance
from src.meshes import occluder_scene
from src.multipath import _planes
from src.ghosts import segments_clear


def _hit_plane(p0, p1, axis, coord):
    """Point where the line p0->p1 crosses the plane axis=coord, and its parameter."""
    den = p1[:, axis] - p0[:, axis]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = (coord - p0[:, axis]) / den
    return p0 + t[:, None] * (p1 - p0), t


def second_order_ghosts_view(room, cfg, radar_pos, facing=True, occlusion=True):
    """Two-bounce ghosts: radar -> plane A -> plane B -> target, direct return.

    Uses the double image to get the path, checks that both bounce points are
    inside the room, and optionally applies incidence weighting and occlusion
    on all three segments. Power has the same factor 2 as the first order
    (the two-bounce leg can be outbound or return).
    """
    radar = np.asarray(radar_pos, dtype=float)
    pts, nrm = room["points"], room["normal"]
    scene = occluder_scene(cfg) if occlusion else None
    size = np.array(cfg["room"]["size_m"], dtype=float)
    tol = 1e-6

    def inside(p):
        return ((p >= -tol) & (p <= size + tol)).all(axis=1)

    dist = np.linalg.norm(pts - radar, axis=1)
    dr = np.maximum(dist, 0.1)
    c1 = np.einsum("ij,ij->i", nrm, radar - pts) / np.maximum(dist, 1e-9)
    base = segments_clear(scene, np.tile(radar, (len(pts), 1)), pts)
    if facing:
        base &= c1 > 0

    planes = list(_planes(cfg).values())
    ranges, powers = [], []
    for i, (ax_a, c_a, m_a) in enumerate(planes):
        if abs(radar[ax_a] - c_a) < 1e-6:
            continue                         # radar sits on plane A
        for j, (ax_b, c_b, m_b) in enumerate(planes):
            if i == j:
                continue
            off = (np.abs(pts[:, ax_a] - c_a) > 1e-6) & (np.abs(pts[:, ax_b] - c_b) > 1e-6)
            idx = np.nonzero(base & off)[0]
            if len(idx) == 0:
                continue
            T = pts[idx]
            P = T.copy()
            P[:, ax_b] = 2 * c_b - P[:, ax_b]          # target mirrored over B
            Q = P.copy()
            Q[:, ax_a] = 2 * c_a - Q[:, ax_a]          # ... then over A
            R = np.tile(radar, (len(idx), 1))
            a, ta = _hit_plane(R, Q, ax_a, c_a)
            b, tb = _hit_plane(a, P, ax_b, c_b)
            ok = (ta > 0) & (ta < 1) & (tb > 0) & (tb < 1) & inside(a) & inside(b)
            idx, T, R, a, b, Q = idx[ok], T[ok], R[ok], a[ok], b[ok], Q[ok]
            if len(idx) == 0:
                continue

            w = np.ones(len(idx))
            if facing:
                to_b = b - T
                c2 = (np.einsum("ij,ij->i", nrm[idx], to_b)
                      / np.maximum(np.linalg.norm(to_b, axis=1), 1e-9))
                w = np.sqrt(np.clip(c1[idx], 0, None) * np.clip(c2, 0, None))
                good = w > 0
                idx, T, R, a, b, Q, w = (x[good] for x in (idx, T, R, a, b, Q, w))
            if occlusion and len(idx):
                clear = (segments_clear(scene, R, a) & segments_clear(scene, a, b)
                         & segments_clear(scene, b, T))
                idx, T, R, Q, w = idx[clear], T[clear], R[clear], Q[clear], w[clear]

            d = dr[idx]
            d2 = np.maximum(np.linalg.norm(Q - R, axis=1), 0.1)
            ranges.append((d + d2) / 2.0)
            powers.append(2.0 * room["strength"][idx]
                          * reflectance(m_a) * reflectance(m_b) * w
                          / (d ** 2 * d2 ** 2))
    if not ranges:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ranges), np.concatenate(powers)