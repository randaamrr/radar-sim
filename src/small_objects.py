import copy

import numpy as np

from src.generator import _overlap, WALL_MARGIN_M
from src.surface import scene_samples, radar_view
from src.compare import smoothed_profile

# kind -> typical size (length, width, height) in meters
SMALL_KINDS = {
    "mug":   (0.10, 0.10, 0.10),
    "books": (0.30, 0.20, 0.15),
    "lamp":  (0.20, 0.20, 0.40),
    "box":   (0.25, 0.25, 0.25),
    "plant": (0.30, 0.30, 0.50),
}
SMALL_MATERIALS = ["wood", "metal", "glass"]
EDGES = (0, 1, 2, 3, 4, 6)


def add_small_objects(cfg, n, seed):
    """Copy of cfg with n small floor objects in free spots (reproducible)."""
    out = copy.deepcopy(cfg)
    rng = np.random.default_rng(seed)
    L, W, H = out["room"]["size_m"]
    placed = [(f["pos_m"][0], f["pos_m"][1], f["size_m"][0], f["size_m"][1])
              for f in out["furniture"]]
    kinds = list(SMALL_KINDS)
    for i in range(n):
        kind = str(rng.choice(kinds))
        scale = rng.uniform(0.8, 1.2, size=3)
        w, d, h = (round(float(s * k), 2) for s, k in zip(SMALL_KINDS[kind], scale))
        if rng.random() < 0.5:
            w, d = d, w
        material = str(rng.choice(SMALL_MATERIALS))
        for _ in range(200):
            x = round(float(rng.uniform(WALL_MARGIN_M, L - w - WALL_MARGIN_M)), 2)
            y = round(float(rng.uniform(WALL_MARGIN_M, W - d - WALL_MARGIN_M)), 2)
            if not any(_overlap((x, y, w, d), p) for p in placed):
                placed.append((x, y, w, d))
                out["furniture"].append({
                    "name": f"small_{kind}_{i}",
                    "pos_m": [x, y],
                    "size_m": [w, d, h],
                    "material": material,
                })
                break
    return out


def profile_power(room, pos, edges):
    d = np.linalg.norm(room["points"] - np.asarray(pos, dtype=float), axis=1)
    p = room["strength"] / np.maximum(d, 0.1) ** 4
    hist, _ = np.histogram(d, bins=edges, weights=p)
    return hist


def evaluate(cfg_base, cfg_test, radar_positions, resolutions=(0.15, 0.05),
             edges=EDGES):
    """Per radar: relative profile change (smoothed) and signed % power change
    per range window, from cfg_base to cfg_test."""
    base = scene_samples(cfg_base)
    test = scene_samples(cfg_test)
    out = []
    for pos in radar_positions:
        vb = radar_view(base, cfg_base, pos)
        vt = radar_view(test, cfg_test, pos)
        l2 = {}
        for res in resolutions:
            pb = smoothed_profile(vb, pos, res)
            pt = smoothed_profile(vt, pos, res)
            l2[res] = float(np.linalg.norm(pt - pb) / np.linalg.norm(pb))
        wb = profile_power(vb, pos, edges)
        wt = profile_power(vt, pos, edges)
        safe = np.where(wb > 0, wb, 1.0)
        change = np.where(wb > 0, 100.0 * (wt - wb) / safe, 0.0)
        out.append({"l2": l2, "window_change_pct": change})
    return out


if __name__ == "__main__":
    from src.scenario import load_scenario

    N_OBJECTS, SEEDS = 6, range(5)
    cfg = load_scenario("scenarios/room_basic.yaml")
    cfg["room"]["grid_step_m"] = 0.05          # must resolve 10 cm objects
    names = [r["name"] for r in cfg["radars"]]
    radars = [r["pos_m"] for r in cfg["radars"]]

    nobig = copy.deepcopy(cfg)
    nobig["furniture"] = []
    big = evaluate(cfg, nobig, radars)

    runs = [evaluate(cfg, add_small_objects(cfg, N_OBJECTS, s), radars)
            for s in SEEDS]

    print(f"grid step 0.05 m, {N_OBJECTS} small objects, "
          f"mean over {len(list(SEEDS))} placements and {len(radars)} radars")
    print(f"{'resolution':>11} {'remove big furniture':>22} {'add small objects':>19}")
    for res in (0.15, 0.05):
        b = np.mean([x["l2"][res] for x in big])
        s = np.mean([x["l2"][res] for run in runs for x in run])
        print(f"{res*100:>8.0f} cm {b:>22.3f} {s:>19.3f}")

    print("\nsmall objects: signed power change by range window (%), mean over placements")
    print(f"{'':>12}" + "".join(f"{lo}-{hi} m".rjust(10)
                                for lo, hi in zip(EDGES[:-1], EDGES[1:])))
    for k, name in enumerate(names):
        m = np.mean([run[k]["window_change_pct"] for run in runs], axis=0)
        print(f"{name:>12}" + "".join(f"{v:>10.1f}" for v in m))