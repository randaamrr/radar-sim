import numpy as np

from src.materials import reflectance


def _planes(cfg):
    """Room boundary planes: name -> (axis, coordinate, material)."""
    L, W, H = cfg["room"]["size_m"]
    s = cfg["surfaces"]
    return {
        "wall_x0": (0, 0.0, s["walls"]),
        "wall_x1": (0, L, s["walls"]),
        "wall_y0": (1, 0.0, s["walls"]),
        "wall_y1": (1, W, s["walls"]),
        "floor": (2, 0.0, s["floor"]),
        "ceiling": (2, H, s["ceiling"]),
    }


def mirror(pos, axis, coord):
    """Image of a position across the plane axis=coord."""
    img = np.array(pos, dtype=float)
    img[axis] = 2.0 * coord - img[axis]
    return img


def first_order_ghosts(room, cfg, radar_pos):
    """One-bounce multipath via the image method.

    Returns (apparent_range_m, power) arrays, one entry per
    (scatterer, bounce plane) pair. Power uses the same model as the direct
    path (strength / (L_out^2 * L_in^2)), times plane reflectance, times 2
    because the bounce can be on the outbound or the return leg.
    Occlusion is ignored here (Yasmine's track).
    """
    radar = np.asarray(radar_pos, dtype=float)
    pts = room["points"]
    d = np.maximum(np.linalg.norm(pts - radar, axis=1), 0.1)

    ranges, powers = [], []
    for axis, coord, mat in _planes(cfg).values():
        if abs(radar[axis] - coord) < 1e-6:
            continue  # radar is mounted on this plane
        img = mirror(radar, axis, coord)
        d_img = np.maximum(np.linalg.norm(pts - img, axis=1), 0.1)
        keep = np.abs(pts[:, axis] - coord) > 1e-6  # skip points on the plane
        apparent = (d + d_img) / 2.0
        p = 2.0 * room["strength"] * reflectance(mat) / (d ** 2 * d_img ** 2)
        ranges.append(apparent[keep])
        powers.append(p[keep])
    return np.concatenate(ranges), np.concatenate(powers)


def _smooth(hist, range_res_m, bin_m):
    sigma = max(range_res_m / bin_m / 2.355, 0.5)
    half = int(4 * sigma)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return np.convolve(hist, k / k.sum(), mode="same")


def direct_and_ghost_profiles(room, cfg, radar_pos, range_res_m=0.15,
                              r_max=10.0, bin_m=0.01):
    """Range profiles (direct, ghost) blurred to the radar range resolution."""
    radar = np.asarray(radar_pos, dtype=float)
    d = np.maximum(np.linalg.norm(room["points"] - radar, axis=1), 0.1)
    edges = np.arange(0.0, r_max + bin_m, bin_m)
    direct, _ = np.histogram(d, bins=edges, weights=room["strength"] / d ** 4)
    gr, gp = first_order_ghosts(room, cfg, radar_pos)
    ghost, _ = np.histogram(gr, bins=edges, weights=gp)
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, _smooth(direct, range_res_m, bin_m), _smooth(ghost, range_res_m, bin_m)


def ghost_to_direct_db(room, cfg, radar_pos):
    """Total ghost power relative to total direct power, in dB."""
    radar = np.asarray(radar_pos, dtype=float)
    d = np.maximum(np.linalg.norm(room["points"] - radar, axis=1), 0.1)
    direct = (room["strength"] / d ** 4).sum()
    _, gp = first_order_ghosts(room, cfg, radar_pos)
    return float(10 * np.log10(gp.sum() / direct))

def second_order_ghost_db(room, cfg, radar_pos):
    """Ghost power from two bounces on distinct planes, relative to direct.

    Round-trip path radar -> plane A -> plane B -> target -> radar is
    approximated with the double image; power gets both reflectances.
    Counts each ordered pair once (a lower bound on the total).
    """
    radar = np.asarray(radar_pos, dtype=float)
    pts = room["points"]
    d = np.maximum(np.linalg.norm(pts - radar, axis=1), 0.1)
    direct = (room["strength"] / d ** 4).sum()

    planes = list(_planes(cfg).values())
    total = 0.0
    for i, (ax_a, c_a, m_a) in enumerate(planes):
        for j, (ax_b, c_b, m_b) in enumerate(planes):
            if i == j:
                continue
            img = mirror(mirror(radar, ax_a, c_a), ax_b, c_b)
            d2 = np.maximum(np.linalg.norm(pts - img, axis=1), 0.1)
            p = (room["strength"] * reflectance(m_a) * reflectance(m_b)
                 / (d ** 2 * d2 ** 2))
            total += p.sum()
    return float(10 * np.log10(total / direct))

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.scenario import load_scenario
    from src.room import build_room
    from src.materials import apply_materials

    cfg = load_scenario("scenarios/room_basic.yaml")
    room = apply_materials(build_room(cfg))
    Path("output").mkdir(exist_ok=True)

    n = len(cfg["radars"])
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(7, 3.5 * n))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, cfg["radars"]):
        db = ghost_to_direct_db(room, cfg, r["pos_m"])
        print(f"{r['name']}: ghost/direct total power = {db:.1f} dB")
        db2 = second_order_ghost_db(room, cfg, r["pos_m"])
        print(f"{r['name']}: 2nd-order ghost/direct = {db2:.1f} dB")
        x, direct, ghost = direct_and_ghost_profiles(room, cfg, r["pos_m"])
        peak = direct.max()
        ax.semilogy(x, np.maximum(direct / peak, 1e-8), label="direct")
        ax.semilogy(x, np.maximum(ghost / peak, 1e-8), label="multipath (1st order)")
        ax.set_ylim(1e-6, 2)
        ax.set_title(r["name"])
        ax.set_ylabel("power (normalized)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("apparent range (m)")
    fig.tight_layout()
    fig.savefig("output/multipath.png", dpi=150)
    print("saved output/multipath.png")